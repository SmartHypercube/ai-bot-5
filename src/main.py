import aiohttp
import asyncio
import base64
import hashlib
import msgpack
import pathlib
import re
import secrets
import sqlite3
import sys
import traceback
import zstandard

SHORT_MESSAGE_LENGTH_LIMIT = 200  # in Unicode code units
MESSAGE_LENGTH_LIMIT = 4096  # in Unicode code units
DEVELOPER = 164648654

pastebin = pathlib.Path('/etc/nginx/sites/ai-bot/public')
db = sqlite3.connect('var/db.sqlite3', isolation_level=None)
session = None  # will be set in main


def serialize_fast(obj):
    return zstandard.compress(msgpack.packb(obj), 3)  # type: ignore


def serialize_slow(obj):
    return zstandard.compress(msgpack.packb(obj), 19)  # type: ignore


def deserialize(data):
    return msgpack.unpackb(zstandard.decompress(data))


def init_db():
    db.execute('pragma journal_mode=wal')
    db.execute('pragma synchronous=normal')
    db.execute('create table config (key text primary key, value)')
    db.execute('create table message (chat_id integer not null, message_id integer not null, data blob not null, primary key (chat_id, message_id))')
    db.execute('create table chat (chat_id integer primary key, data blob not null)')
    db.execute('insert into config (key, value) values (?, ?)', ['telegram_token', input('Telegram bot token: ').strip()])
    db.execute('insert into config (key, value) values (?, ?)', ['openai_api_key', input('OpenAI API key: ').strip()])
    db.execute('insert into config (key, value) values (?, ?)', ['gemini_api_key', input('Gemini API key: ').strip()])
    print('输入白名单的用户 ID，每行一个，# 后面的内容是注释，以空行结束：')
    print(f'> {DEVELOPER}  # 源代码中的 DEVELOPER 常量')
    lines = [f'{DEVELOPER}  # 源代码中的 DEVELOPER 常量']
    while True:
        line = input('> ').strip()
        if not line:
            break
        lines.append(line)
    db.execute('insert into config (key, value) values (?, ?)', ['whitelist', '\n'.join(lines) + '\n'])
    print('输入严格隐私模式的用户 ID，每行一个，# 后面的内容是注释，以空行结束：')
    lines = []
    while True:
        line = input('> ').strip()
        if not line:
            break
        lines.append(line)
    db.execute('insert into config (key, value) values (?, ?)', ['strict_privacy', '\n'.join(lines) + '\n'])
    db.execute('insert into config (key, value) values (?, ?)', ['safety_identifier_salt', secrets.token_urlsafe(16)])


def get_offset():
    if row := db.execute('select value from config where key = ?', ['offset']).fetchone():
        return row[0]
    return 0


def set_offset(offset):
    db.execute('insert into config (key, value) values (?, ?) on conflict do update set value = excluded.value', ['offset', offset])


def escape_html(text):
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


async def telegram_api(_method, **kwargs):
    [token] = db.execute('select value from config where key = ?', ['telegram_token']).fetchone()
    async with session.post(f'https://api.telegram.org/bot{token}/{_method}', json=kwargs) as response:
        if response.status != 200:
            try:
                description = (await response.json())['description']
            except Exception:
                description = None
            if _method == 'editMessageText' and description == 'Bad Request: message is not modified: specified new message content and reply markup are exactly the same as a current content and reply markup of the message':
                return None
            raise RuntimeError(f'Telegram API error: {response.status} {await response.text()}')
        response = await response.json()
    if not response['ok']:
        raise RuntimeError(f'Telegram API error: {response["description"]}')
    return response['result']


async def telegram_send_text(chat_id, text, reply_to_message_id=None, markdown=True):
    data = {}
    if row := db.execute('select data from chat where chat_id = ?', [chat_id]).fetchone():
        data = deserialize(row[0])
    if data.get('fold', False):
        if len(text) <= SHORT_MESSAGE_LENGTH_LIMIT:
            kwargs = {'text': text}
        else:
            pastebin_path = secrets.token_urlsafe(8)
            (pastebin/pastebin_path).write_text(text)
            if markdown:
                url = f'https://bg46.org/q#{pastebin_path}'
            else:
                url = f'https://bg46.org/{pastebin_path}'
            if len(text) > MESSAGE_LENGTH_LIMIT - 1 - len(url):
                tail = '\n\n（已截断，点击链接查看完整内容）'
                text = text[:MESSAGE_LENGTH_LIMIT - 1 - len(url) - len(tail)] + tail
            kwargs = {
                'text': '<blockquote expandable>' + escape_html(text) + '</blockquote>\n' + url,
                'parse_mode': 'HTML',
            }
        if reply_to_message_id is not None:
            kwargs['reply_parameters'] = {'message_id': reply_to_message_id}
        result = await telegram_api(
            'sendMessage',
            chat_id=chat_id,
            link_preview_options={'is_disabled': True},
            **kwargs,
        )
        return [result['message_id']]
    else:
        message_ids = []
        while text:
            part = text[:MESSAGE_LENGTH_LIMIT]
            text = text[MESSAGE_LENGTH_LIMIT:]
            kwargs = {'text': part}
            if reply_to_message_id is not None:
                kwargs['reply_parameters'] = {'message_id': reply_to_message_id}
            result = await telegram_api(
                'sendMessage',
                chat_id=chat_id,
                link_preview_options={'is_disabled': True},
                **kwargs,
            )
            message_ids.append(result['message_id'])
            reply_to_message_id = result['message_id']
        return message_ids


async def telegram_get_file(file_id):
    file_path = (await telegram_api('getFile', file_id=file_id))['file_path']
    [token] = db.execute('select value from config where key = ?', ['telegram_token']).fetchone()
    async with session.get(f'https://api.telegram.org/file/bot{token}/{file_path}') as response:
        if response.status != 200:
            raise RuntimeError(f'Telegram file download error: {response.status} {await response.text()}')
        return await response.read()


def get_whitelist():
    result = set()
    if row := db.execute('select value from config where key = ?', ['whitelist']).fetchone():
        for line in row[0].splitlines():
            if line := line.split('#', 1)[0].strip():
                result.add(int(line))
    return result


def get_strict_privacy_users():
    result = set()
    if row := db.execute('select value from config where key = ?', ['strict_privacy']).fetchone():
        for line in row[0].splitlines():
            if line := line.split('#', 1)[0].strip():
                result.add(int(line))
    return result


def parse_prefix(prefix):
    models = []
    for part in prefix.split(','):
        part = part.strip()
        if not part:
            raise ValueError('Empty model specification')

        s, *args = part.split('+')
        s = s.strip()
        model = {
            '5': 'gpt-5',
            '5m': 'gpt-5-mini',
            '5n': 'gpt-5-nano',
            '25p': 'gemini-2.5-pro',
            '25f': 'gemini-2.5-flash',
            '25l': 'gemini-2.5-flash-lite',
        }.get(s, s)
        if not model:
            raise ValueError(f'Must specify a model')
        if not model.startswith(('gpt-', 'gemini-')):
            raise ValueError(f'Model name must start with gpt- or gemini-: {model}')
        if not re.fullmatch(r'(?a)[\w.-]+', model):
            raise ValueError(f'Invalid model name: {model}')
        model = {'model': model}

        model_type = model['model'].split('-', 1)[0]
        for arg in args:
            match arg[0], model_type:
                case 'r', 'gpt':
                    s = arg[1:].strip()
                    if not s:
                        raise ValueError('Missing reasoning effort')
                    for i in ['low', 'medium', 'high', 'minimal']:
                        if i.startswith(s.lower()):
                            model['reasoning'] = i
                            break
                    else:
                        raise ValueError(f'Unknown reasoning effort: {arg[1:]}')
                case 'r', 'gemini':
                    s = arg[1:].strip()
                    if not s:
                        raise ValueError('Missing reasoning effort')
                    if re.fullmatch(r'(?a)\d+', s):
                        model['reasoning'] = int(s)
                    else:
                        for i in ['low', 'medium', 'high', 'minimal', 'none', 'dynamic']:
                            if i.startswith(s.lower()):
                                model['reasoning'] = i
                                break
                        else:
                            raise ValueError(f'Unknown reasoning effort: {arg[1:]}')
                case 'v', 'gpt':
                    s = arg[1:].strip()
                    if not s:
                        raise ValueError('Missing verbosity')
                    for i in ['low', 'medium', 'high']:
                        if i.startswith(s.lower()):
                            model['verbosity'] = i
                            break
                    else:
                        raise ValueError(f'Unknown verbosity: {arg[1:]}')
                case 't', _:
                    s = arg[1:].strip()
                    if len(set(s)) != len(s):
                        raise ValueError(f'Duplicate tool options: {arg[1:]}')
                    if set(s) - {'s', 'c'}:
                        raise ValueError(f'Unknown tool options: {arg[1:]}')
                    if s:
                        model['tools'] = s
                case 's', _:
                    s = arg[1:].strip()
                    if s:
                        model['system'] = s
                case _:
                    raise ValueError(f'Unknown model argument: +{arg}')
        models.append(model)

    if not models:
        raise ValueError('Must specify at least one model')
    return models


def format_prefix(models):
    parts = []
    for model in models:
        s = model['model']
        s = {
            'gpt-5': '5',
            'gpt-5-mini': '5m',
            'gpt-5-nano': '5n',
            'gemini-2.5-pro': '25p',
            'gemini-2.5-flash': '25f',
            'gemini-2.5-flash-lite': '25l',
        }.get(s, s)
        if 'reasoning' in model:
            if isinstance(model['reasoning'], int):
                s += '+r' + str(model['reasoning'])
            else:
                s += '+r' + {'low': 'l', 'medium': 'm', 'high': 'h', 'minimal': 'min', 'none': 'n', 'dynamic': 'd'}[model['reasoning']]
        if 'verbosity' in model:
            s += '+v' + {'low': 'l', 'medium': 'm', 'high': 'h'}[model['verbosity']]
        if 'tools' in model:
            s += '+t' + model['tools']
        if 'system' in model:
            if '+' in model['system'] or ',' in model['system'] or '$' in model['system']:
                return None
            s += '+s' + model['system']
        parts.append(s)
    return ','.join(parts)


def gemini_reasoning_to_number(model, reasoning):
    if isinstance(reasoning, int):
        return reasoning
    match model, reasoning:
        case 'gemini-2.5-pro', 'high':
            return 32768
        case 'gemini-2.5-flash-lite', 'minimal':
            return 512
        case _:
            return {'none': 0, 'minimal': 128, 'low': 1024, 'medium': 8192, 'high': 24576, 'dynamic': -1}[reasoning]


async def render_select_model_state(state, chat_id, message_id=None):
    kwargs = {
        'chat_id': chat_id,
        'parse_mode': 'HTML',
        'link_preview_options': {'is_disabled': True},
    }
    match state:
        case {'step': 'model'}:
            kwargs['text'] = '基础模型：请选择'
            kwargs['reply_markup'] = {
                'inline_keyboard': [
                    [
                        {'text': 'gpt-5', 'callback_data': 'm/gpt-5'},
                        {'text': 'gpt-5-mini', 'callback_data': 'm/gpt-5-mini'},
                        {'text': 'gpt-5-nano', 'callback_data': 'm/gpt-5-nano'},
                    ],
                    [
                        {'text': 'gemini-2.5-pro', 'callback_data': 'm/gemini-2.5-pro'},
                        {'text': 'gemini-2.5-flash', 'callback_data': 'm/gemini-2.5-flash'},
                    ],
                    [
                        {'text': 'gemini-2.5-flash-lite', 'callback_data': 'm/gemini-2.5-flash-lite'},
                    ],
                    [
                        {'text': '其他（不一定兼容）', 'callback_data': 'm/_other'},
                    ],
                ],
            }
        case {'step': 'model-input'}:
            kwargs['text'] = '请回复 OpenAI 或 Gemini 模型名称'
            kwargs['reply_markup'] = {'force_reply': True, 'input_field_placeholder': 'gpt-... / gemini-...'}
        case {'step': 'model-re-input'}:
            kwargs['text'] = '模型名称必须以 gpt- 或 gemini- 开头，请重新输入'
            kwargs['reply_markup'] = {'force_reply': True, 'input_field_placeholder': 'gpt-... / gemini-...'}
        case {'step': 'system-input'}:
            kwargs['text'] = '请回复系统提示'
            kwargs['reply_markup'] = {'force_reply': True, 'input_field_placeholder': '你是一个...'}
        case {'step': 'ready' | 'used' | 'invalid'}:
            lines = ['基础模型：' + escape_html(state['model'])]
            model_type = state['model'].split('-', 1)[0]
            if model_type == 'gpt':
                lines.append('推理努力：' + state.get('reasoning', '不指定（默认 medium）'))
                lines.append('输出长度：' + state.get('verbosity', '不指定（默认 medium）'))
            elif model_type == 'gemini':
                if 'reasoning' not in state:
                    lines.append('推理努力：不指定（默认自动决定）')
                elif isinstance(state['reasoning'], int):
                    lines.append(f'推理努力：{state["reasoning"]}')
                else:
                    v = gemini_reasoning_to_number(state['model'], state['reasoning'])
                    s = {'none': f'无（{v}）', 'minimal': f'最低（{v}）', 'low': f'低（{v}）', 'medium': f'中（{v}）', 'high': f'高（{v}）', 'dynamic': '自动决定'}[state['reasoning']]
                    lines.append(f'推理努力：{s}')
            lines.append('网页搜索：' + ('开' if 's' in state.get('tools', '') else '关'))
            lines.append('运行代码：' + ('开' if 'c' in state.get('tools', '') else '关'))
            if 'system' in state:
                lines.append('系统提示：')
                lines.append('<blockquote expandable>' + escape_html(state['system']) + '</blockquote>')
            else:
                lines.append('系统提示：无')
            lines.append('')
            if state['step'] == 'invalid':
                lines.append(f'错误：{state["error"]}')
            else:
                lines.append('回复这条消息开始对话')
                if prefix := format_prefix([state]):
                    lines.append('也可以发送以 <code>' + escape_html(prefix) + '$</code>（点击复制）开头的新消息开始对话')
            kwargs['text'] = '\n'.join(lines)
            keyboard = []
            keyboard.append([{'text': '修改基础模型', 'callback_data': 'm/_change'}])
            if model_type == 'gpt':
                keyboard.append([
                    {'text': '推理：', 'callback_data': 'r/'},
                    {'text': 'minimal', 'callback_data': 'r/minimal'},
                    {'text': 'low', 'callback_data': 'r/low'},
                    {'text': 'medium', 'callback_data': 'r/medium'},
                    {'text': 'high', 'callback_data': 'r/high'},
                ])
                keyboard.append([
                    {'text': '长度：', 'callback_data': 'v/'},
                    {'text': 'low', 'callback_data': 'v/low'},
                    {'text': 'medium', 'callback_data': 'v/medium'},
                    {'text': 'high', 'callback_data': 'v/high'},
                ])
            elif model_type == 'gemini':
                l = [
                    {'text': '推理：', 'callback_data': 'r/'},
                    {'text': '自动', 'callback_data': 'r/dynamic'},
                ]
                if state['model'] != 'gemini-2.5-pro':
                    l.append({'text': '无', 'callback_data': 'r/none'})
                l += [
                    {'text': '最低', 'callback_data': 'r/minimal'},
                    {'text': '低', 'callback_data': 'r/low'},
                    {'text': '中', 'callback_data': 'r/medium'},
                    {'text': '高', 'callback_data': 'r/high'},
                ]
                keyboard.append(l)
            keyboard.append([
                {
                    True: {'text': '禁用网页搜索', 'callback_data': 'tn/s'},
                    False: {'text': '启用网页搜索', 'callback_data': 'ty/s'},
                }['s' in state.get('tools', '')],
                {
                    True: {'text': '禁用运行代码', 'callback_data': 'tn/c'},
                    False: {'text': '启用运行代码', 'callback_data': 'ty/c'},
                }['c' in state.get('tools', '')],
            ])
            keyboard.append([
                {'text': '清除系统提示', 'callback_data': 's/'},
                {'text': '修改系统提示', 'callback_data': 's/_change'},
            ])
            keyboard.append([
                {'text': '复制', 'callback_data': 'copy'},
                {'text': '设为默认模型', 'callback_data': 'default'},
            ])
            kwargs['reply_markup'] = {'inline_keyboard': keyboard}
    if message_id is None:
        result = await telegram_api('sendMessage', **kwargs)
        message_id = result['message_id']
    else:
        await telegram_api('editMessageText', message_id=message_id, **kwargs)
    db.execute('insert into message (chat_id, message_id, data) values (?, ?, ?) on conflict do update set data = excluded.data', [chat_id, message_id, serialize_fast(state)])


def select_model_after_change_model(state):
    model_type = state['model'].split('-', 1)[0]
    if model_type == 'gpt':
        match state.get('reasoning', None):
            case 'dynamic':
                del state['reasoning']
            case 'none':
                state['reasoning'] = 'minimal'
    elif model_type == 'gemini':
        state.pop('verbosity', None)


def select_model_check_invalid(state):
    if state['step'] not in {'ready', 'invalid'}:
        return
    model_type = state['model'].split('-', 1)[0]
    error = None
    if model_type == 'gpt' and state.get('reasoning', None) == 'minimal' and state.get('tools', ''):
        error = '推理努力为 minimal 时不支持使用工具'
    if error is None:
        state['step'] = 'ready'
        state.pop('error', None)
    else:
        state['step'] = 'invalid'
        state['error'] = error


async def handle_select_model_callback_query(state, callback_query):
    assert state['type'] == 'select_model'
    chat_id = callback_query['message']['chat']['id']
    message_id = callback_query['message']['message_id']
    data = callback_query['data']
    if state['step'] == 'used':
        message_id = None
        state['step'] = 'ready'
    match data.split('/', 1):
        case ['m', '_other']:
            state['step'] = 'model-input'
            await render_select_model_state(state, chat_id)
        case ['m', '_change']:
            state['step'] = 'model'
            await render_select_model_state(state, chat_id, message_id)
        case ['m', model]:
            if not model.startswith(('gpt-', 'gemini-')):
                raise ValueError(f'Model name must start with gpt- or gemini-: {model}')
            if not re.fullmatch(r'(?a)[\w.-]+', model):
                raise ValueError(f'Invalid model name: {model}')
            state['model'] = model
            state['step'] = 'ready'
            select_model_after_change_model(state)
            select_model_check_invalid(state)
            await render_select_model_state(state, chat_id, message_id)
        case ['r', '']:
            state.pop('reasoning', None)
            select_model_check_invalid(state)
            await render_select_model_state(state, chat_id, message_id)
        case ['r', reasoning]:
            state['reasoning'] = reasoning
            select_model_check_invalid(state)
            await render_select_model_state(state, chat_id, message_id)
        case ['v', '']:
            state.pop('verbosity', None)
            select_model_check_invalid(state)
            await render_select_model_state(state, chat_id, message_id)
        case ['v', verbosity]:
            state['verbosity'] = verbosity
            select_model_check_invalid(state)
            await render_select_model_state(state, chat_id, message_id)
        case ['tn', tool]:
            state['tools'] = state.get('tools', '').replace(tool, '')
            if not state['tools']:
                del state['tools']
            select_model_check_invalid(state)
            await render_select_model_state(state, chat_id, message_id)
        case ['ty', tool]:
            state['tools'] = state.get('tools', '').replace(tool, '') + tool
            select_model_check_invalid(state)
            await render_select_model_state(state, chat_id, message_id)
        case ['s', '']:
            state.pop('system', None)
            select_model_check_invalid(state)
            await render_select_model_state(state, chat_id, message_id)
        case ['s', '_change']:
            state['step'] = 'system-input'
            await render_select_model_state(state, chat_id)
        case ['copy']:
            await render_select_model_state(state, chat_id)
        case ['default']:
            if state['step'] not in {'ready', 'used'}:
                return {
                    'text': '模型选择尚未完成',
                    'show_alert': True,
                }
            data = {}
            if row := db.execute('select data from chat where chat_id = ?', [chat_id]).fetchone():
                data = deserialize(row[0])
            data['default_models'] = [{k: v for k, v in state.items() if k in {'model', 'reasoning', 'verbosity', 'tools', 'system'}}]
            db.execute('insert into chat (chat_id, data) values (?, ?) on conflict do update set data = excluded.data', [chat_id, serialize_fast(data)])
            state['step'] = 'used'
            db.execute('insert into message (chat_id, message_id, data) values (?, ?, ?) on conflict do update set data = excluded.data', [chat_id, message_id, serialize_fast(state)])
            await telegram_api(
                'sendMessage',
                chat_id=chat_id,
                text='已将这个模型设置为本聊天的默认模型，直接发送新消息即可使用。发送 /unset_default 清除默认模型。\n\n提示：可以创建多个只包含你和 bot 的群，每个群可以设置不同的默认模型。',
                reply_parameters={'message_id': callback_query['message']['message_id']},
            )
        case _:
            raise ValueError('Invalid callback query data')


async def handle_select_model_message(state, message):
    assert state['type'] == 'select_model'
    chat_id = message['chat']['id']
    text = message['text']
    match state['step']:
        case 'model-input' | 'model-re-input':
            model = text.strip().lower()
            if model.startswith(('gpt-', 'gemini-')) and re.fullmatch(r'(?a)[\w.-]+', model):
                state['model'] = model
                state['step'] = 'ready'
                select_model_after_change_model(state)
                select_model_check_invalid(state)
            else:
                state['step'] = 'model-re-input'
            await render_select_model_state(state, chat_id)
        case 'system-input':
            state['system'] = text.strip()
            state['step'] = 'ready'
            await render_select_model_state(state, chat_id)
        case _:
            raise ValueError('Invalid reply message')


async def handle_message(message):
    chat_id = message['chat']['id']
    from_id = message['from']['id']
    strict_privacy = from_id in get_strict_privacy_users()

    try:
        if message.get('text', '').startswith('/select_model'):
            state = {'type': 'select_model', 'step': 'model'}
            await render_select_model_state(state, chat_id)
            return
        if message.get('text', '').startswith('/unset_default'):
            data = {}
            if row := db.execute('select data from chat where chat_id = ?', [chat_id]).fetchone():
                data = deserialize(row[0])
            if 'default_models' in data:
                del data['default_models']
                db.execute('insert into chat (chat_id, data) values (?, ?) on conflict do update set data = excluded.data', [chat_id, serialize_fast(data)])
                await telegram_send_text(chat_id, '已清除默认模型。发送 /select_model 选择模型。')
            else:
                await telegram_send_text(chat_id, '当前没有设置默认模型。发送 /select_model 选择模型。')
            return
        if message.get('text', '').startswith('/enable_fold'):
            data = {}
            if row := db.execute('select data from chat where chat_id = ?', [chat_id]).fetchone():
                data = deserialize(row[0])
            if not data.get('fold', False):
                data['fold'] = True
                db.execute('insert into chat (chat_id, data) values (?, ?) on conflict do update set data = excluded.data', [chat_id, serialize_fast(data)])
            await telegram_send_text(chat_id, '已启用长输出折叠功能。发送 /disable_fold 禁用折叠功能。')
            return
        if message.get('text', '').startswith('/disable_fold'):
            data = {}
            if row := db.execute('select data from chat where chat_id = ?', [chat_id]).fetchone():
                data = deserialize(row[0])
            if data.get('fold', False):
                data['fold'] = False
                db.execute('insert into chat (chat_id, data) values (?, ?) on conflict do update set data = excluded.data', [chat_id, serialize_fast(data)])
            await telegram_send_text(chat_id, '已禁用长输出折叠功能。发送 /enable_fold 启用折叠功能。')
            return

        message_id = message['message_id']
        if 'media_group_id' in message:
            await telegram_send_text(chat_id, '只支持每次发送一张图片或一个文件。', reply_to_message_id=message_id)
            return

        reply_file = None
        history = None
        models = None
        if 'reply_to_message' in message:
            reply_to_message_id = message['reply_to_message']['message_id']
            if (row := db.execute('select data from message where chat_id = ? and message_id = ?', [message['chat']['id'], reply_to_message_id]).fetchone()) is None:
                await telegram_send_text(chat_id, '只支持回复 AI 发送的消息，或者你发送的图片或文件。', reply_to_message_id=message_id)
                return
            match deserialize(row[0]):
                case {'type': 'history', 'history': history, 'models': models}:
                    pass
                case {'type': 'file'} as reply_file:
                    pass
                case {'type': 'select_model', 'step': 'ready' | 'used'} as state:
                    if state['step'] == 'ready':
                        state['step'] = 'used'
                        db.execute('insert into message (chat_id, message_id, data) values (?, ?, ?) on conflict do update set data = excluded.data', [chat_id, reply_to_message_id, serialize_fast(state)])
                    models = [{k: v for k, v in state.items() if k in {'model', 'reasoning', 'verbosity', 'tools', 'system'}}]
                case {'type': 'select_model', 'step': 'model-input' | 'model-re-input' | 'system-input'} as state:
                    await handle_select_model_message(state, message)
                    return
                case {'type': 'select_model', 'step': _}:
                    await telegram_send_text(chat_id, '模型选择尚未完成。', reply_to_message_id=message_id)
                    return
                case _:
                    assert False

        if reply_file and ('photo' in message or 'document' in message):
            await telegram_send_text(chat_id, '回复你发送的图片或文件时只支持发送文字消息。', reply_to_message_id=message_id)
            return

        file = None
        if 'photo' in message:
            file_id = max(message['photo'], key=lambda i: i['file_size'])['file_id']
            file = {
                'type': 'file',
                'mime_type': 'image/jpeg',
                'data': await telegram_get_file(file_id),
            }
        elif 'document' in message:
            mime_type = message['document'].get('mime_type', '')
            if not (mime_type.startswith(('image/', 'text/')) or mime_type == 'application/pdf'):
                await telegram_send_text(chat_id, '只支持发送文字消息、图片、文本文件或 PDF 文件。', reply_to_message_id=message_id)
                return
            file_id = message['document']['file_id']
            file = {
                'type': 'file',
                'mime_type': mime_type,
                'data': await telegram_get_file(file_id),
            }
            if 'file_name' in message['document']:
                file['name'] = message['document']['file_name']
        if file:
            db.execute('insert into message (chat_id, message_id, data) values (?, ?, ?)', [chat_id, message_id, serialize_fast(file)])

        text = message.get('text', '') + message.get('caption', '')
        if '$' in text.split('\n', 1)[0]:
            prefix, text = text.split('$', 1)
            models = parse_prefix(prefix)
        if not text.strip():
            text = None

        if models is None:
            if row := db.execute('select data from chat where chat_id = ?', [chat_id]).fetchone():
                data = deserialize(row[0])
                if 'default_models' in data:
                    models = data['default_models']
        if models is None:
            await telegram_send_text(chat_id, '发送 /select_model 选择模型。', reply_to_message_id=message_id)
            return

        if not text and not file:
            await telegram_send_text(chat_id, '只支持发送文字消息、图片、文本文件或 PDF 文件。', reply_to_message_id=message_id)
            return
        file = file or reply_file

        [safety_identifier_salt] = db.execute('select value from config where key = ?', ['safety_identifier_salt']).fetchone()
        if len(safety_identifier_salt) < 16:
            raise ValueError('safety_identifier_salt too short')
        safety_identifier = hashlib.sha256(f'{safety_identifier_salt},{from_id}'.encode()).hexdigest()
        for i, model in enumerate(models):
            if len(models) == 1:
                i = None
            if model['model'].startswith('gpt-'):
                asyncio.create_task(complete_and_reply_gpt(chat_id, message_id, i, model, history, file, text, safety_identifier, strict_privacy))
            elif model['model'].startswith('gemini-'):
                asyncio.create_task(complete_and_reply_gemini(chat_id, message_id, i, model, history, file, text, safety_identifier, strict_privacy))
            else:
                assert False

    except Exception:
        if strict_privacy:
            try:
                text = traceback.format_exc()
                await telegram_send_text(chat_id, text, markdown=False)
            except Exception:
                pass
        else:
            text = traceback.format_exc()
            await telegram_send_text(DEVELOPER, text, markdown=False)
            await telegram_send_text(chat_id, '出错了，请重试。')


async def handle_callback_query(callback_query):
    chat_id = callback_query['message']['chat']['id']
    strict_privacy = callback_query['from']['id'] in get_strict_privacy_users()
    kwargs = None
    try:
        message_id = callback_query['message']['message_id']
        row = db.execute('select data from message where chat_id = ? and message_id = ?', [chat_id, message_id]).fetchone()
        match deserialize(row[0]):
            case {'type': 'select_model'} as state:
                kwargs = await handle_select_model_callback_query(state, callback_query)
                return
            case _:
                raise ValueError('Unknown callback query message')
    except Exception:
        if strict_privacy:
            try:
                text = traceback.format_exc()
                await telegram_send_text(chat_id, text, markdown=False)
            except Exception:
                pass
        else:
            text = traceback.format_exc()
            await telegram_send_text(DEVELOPER, text, markdown=False)
            await telegram_send_text(chat_id, '出错了，请重试。')
    finally:
        if kwargs is None:
            kwargs = {}
        await telegram_api(
            'answerCallbackQuery',
            callback_query_id=callback_query['id'],
            **kwargs,
        )


async def complete_and_reply_gpt(chat_id, message_id, model_index, model, history, file, text, safety_identifier, strict_privacy):
    try:
        if 'system' in model:
            assert history is None
            if model['system']:
                history = [{'role': 'system', 'content': [{'type': 'input_text', 'text': model['system']}] }]
            del model['system']
        if history is None:
            history = []

        content = []
        if file:
            if file['mime_type'].startswith('image/'):
                content.append({
                    'type': 'input_image',
                    'image_url': f'data:{file["mime_type"]};base64,' + base64.b64encode(file['data']).decode(),
                    'detail': 'high',
                })
            elif file['mime_type'] == 'application/pdf':
                content.append({
                    'type': 'input_file',
                    'file_data': f'data:{file["mime_type"]};base64,' + base64.b64encode(file['data']).decode(),
                } | ({'filename': file['name']} if 'name' in file else {}))
            elif file['mime_type'].startswith('text/'):
                if text:
                    text = file['data'].decode(errors='replace') + '\n\n' + text
                else:
                    text = file['data'].decode(errors='replace')
                content.append({
                    'type': 'input_text',
                    'text': text,
                })
                text = None
            else:
                assert False
        if text:
            content.append({
                'type': 'input_text',
                'text': text,
            })
        history.append({'role': 'user', 'content': content})

        [openai_api_key] = db.execute('select value from config where key = ?', ['openai_api_key']).fetchone()
        kwargs = {
            'model': model['model'],
            'include': ['reasoning.encrypted_content'],
            'input': history,
            'safety_identifier': safety_identifier,
            'store': False,
        }
        if 'reasoning' in model:
            kwargs['reasoning'] = {'effort': model['reasoning']}
        if 'verbosity' in model:
            kwargs['text'] = {'verbosity': model['verbosity']}
        if 'tools' in model:
            kwargs['tools'] = []
            for i in model['tools']:
                match i:
                    case 's':
                        kwargs['tools'].append({'type': 'web_search', 'search_context_size': 'high'})
                    case 'c':
                        kwargs['tools'].append({'type': 'code_interpreter', 'container': {'type': 'auto'}})
                    case _:
                        raise ValueError(f'Unknown tool option: {i}')

        async with session.post(
            'https://api.openai.com/v1/responses',
            headers={'Authorization': f'Bearer {openai_api_key}'},
            json=kwargs,
            timeout=aiohttp.ClientTimeout(total=3600),
        ) as response:
            if response.status != 200:
                raise RuntimeError(f'OpenAI API error: {response.status} {await response.text()}')
            response = await response.json()
        if response['status'] != 'completed':
            raise RuntimeError(f'OpenAI API error: status {response["status"]}')
        if response['error'] is not None:
            raise RuntimeError(f'OpenAI API error: error {response["error"]}')

        text = ''
        if model_index is not None:
            text += f'（模型 {model_index + 1}）'
        for output in response['output']:
            match output['type']:
                case 'reasoning':
                    history.append(output)
                case 'web_search_call':
                    text += '（网页搜索）'
                case 'code_interpreter_call':
                    text += '（运行代码）'
                case 'message':
                    history.append(output)
                    for content in output['content']:
                        match content['type']:
                            case 'output_text':
                                text += content['text']
                            case type:
                                text += f'（未知内容类型：{type}）'
                case type:
                    text += f'（未知输出类型：{type}）'

        for i in await telegram_send_text(chat_id, text, reply_to_message_id=message_id):
            db.execute('insert into message (chat_id, message_id, data) values (?, ?, ?)', [chat_id, i, serialize_fast({
                'type': 'history',
                'history': history,
                'models': [model],
            })])

    except Exception:
        if strict_privacy:
            try:
                text = traceback.format_exc()
                await telegram_send_text(chat_id, text, markdown=False)
            except Exception:
                pass
        else:
            text = traceback.format_exc()
            await telegram_send_text(DEVELOPER, text, markdown=False)
            await telegram_send_text(chat_id, '出错了，请重试。')


async def complete_and_reply_gemini(chat_id, message_id, model_index, model, history, file, text, safety_identifier, strict_privacy):
    try:
        if history is None:
            history = []

        parts = []
        if file:
            if file['mime_type'].startswith('image/') or file['mime_type'] == 'application/pdf':
                parts.append({
                    'inline_data': {
                        'mime_type': file['mime_type'],
                        'data': base64.b64encode(file['data']).decode(),
                    },
                })
            elif file['mime_type'].startswith('text/'):
                if text:
                    text = file['data'].decode(errors='replace') + '\n\n' + text
                else:
                    text = file['data'].decode(errors='replace')
                parts.append({'text': text})
                text = None
            else:
                assert False
        if text:
            parts.append({'text': text})
        history.append({'role': 'user', 'parts': parts})

        [gemini_api_key] = db.execute('select value from config where key = ?', ['gemini_api_key']).fetchone()
        kwargs = {
            'contents': history,
            'safetySettings': [
                {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'OFF'},
                {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'OFF'},
                {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'OFF'},
                {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'OFF'},
                {'category': 'HARM_CATEGORY_CIVIC_INTEGRITY', 'threshold': 'OFF'},
            ],
        }
        if 'reasoning' in model:
            kwargs['generationConfig'] = {'thinkingConfig': {'thinkingBudget': gemini_reasoning_to_number(model['model'], model['reasoning'])}}
        if 'tools' in model:
            kwargs['tools'] = []
            for i in model['tools']:
                match i:
                    case 's':
                        kwargs['tools'].append({'googleSearch': {}})
                        kwargs['tools'].append({'urlContext': {}})
                    case 'c':
                        kwargs['tools'].append({'codeExecution': {}})
                    case _:
                        raise ValueError(f'Unknown tool option: {i}')
        if model.get('system', ''):
            kwargs['systemInstruction'] = {'parts': [{'text': model['system']}]}

        async with session.post(
            f'https://generativelanguage.googleapis.com/v1beta/models/{model["model"]}:generateContent',
            headers={'x-goog-api-key': gemini_api_key},
            json=kwargs,
            timeout=aiohttp.ClientTimeout(total=3600),
        ) as response:
            if response.status != 200:
                raise RuntimeError(f'Gemini API error: {response.status} {await response.text()}')
            response = await response.json()
        if 'candidates' not in response or len(response['candidates']) != 1:
            raise RuntimeError('Gemini API error: no candidates')

        text = ''
        if model_index is not None:
            text += f'（模型 {model_index + 1}）'
        content = response['candidates'][0]['content']
        for part in content['parts']:
            match part:
                case {'executableCode': _}:
                    text += '（运行代码）'
                case {'codeExecutionResult': _}:
                    pass
                case {'text': s}:
                    text += s
                case _:
                    if len(part) == 1:
                        text += f'（未知内容类型：{next(iter(part))}）'
                    else:
                        text += f'（未知内容类型：{part}）'

        for i in await telegram_send_text(chat_id, text, reply_to_message_id=message_id):
            db.execute('insert into message (chat_id, message_id, data) values (?, ?, ?)', [chat_id, i, serialize_fast({
                'type': 'history',
                'history': history + [content],
                'models': [model],
            })])

    except Exception:
        if strict_privacy:
            try:
                text = traceback.format_exc()
                await telegram_send_text(chat_id, text, markdown=False)
            except Exception:
                pass
        else:
            text = traceback.format_exc()
            await telegram_send_text(DEVELOPER, text, markdown=False)
            await telegram_send_text(chat_id, '出错了，请重试。')


async def main():
    global session
    session = aiohttp.ClientSession(
        cookie_jar=aiohttp.DummyCookieJar(),
        timeout=aiohttp.ClientTimeout(total=30),
    )

    await telegram_api(
        'setMyCommands',
        commands=[
            {'command': 'select_model', 'description': '选择模型'},
            {'command': 'unset_default', 'description': '清除默认模型'},
            {'command': 'enable_fold', 'description': '启用长输出折叠功能'},
            {'command': 'disable_fold', 'description': '禁用长输出折叠功能'},
        ],
    )

    while True:
        try:
            updates = await telegram_api('getUpdates', offset=get_offset(), timeout=10)
        except asyncio.TimeoutError:
            await asyncio.sleep(3)
            continue
        for update in updates:
            match update:
                case {'message': message}:
                    if message['from']['id'] in get_whitelist():
                        asyncio.create_task(handle_message(message))
                case {'callback_query': callback_query}:
                    if callback_query['from']['id'] in get_whitelist():
                        asyncio.create_task(handle_callback_query(callback_query))
            set_offset(update['update_id'] + 1)


if __name__ == '__main__':
    match sys.argv[1:]:
        case []:
            asyncio.run(main())
        case ['init_db']:
            init_db()
        case _:
            print('Unknown command', file=sys.stderr)
            sys.exit(1)
