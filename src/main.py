import aiohttp
import asyncio
import base64
import dataclasses
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
    db.execute('create table search_data (chat_id integer not null, message_id integer not null, text text not null, parent integer, children blob, time integer not null, primary key (chat_id, message_id))')
    db.execute('create table message_group (chat_id integer not null, message_id integer not null, group_id integer not null, primary key (chat_id, message_id))')

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
        return await telegram_api(
            'sendMessage',
            chat_id=chat_id,
            link_preview_options={'is_disabled': True},
            **kwargs,
        )
    else:
        messages = []
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
            messages.append(result)
            reply_to_message_id = result['message_id']
        for i in messages[1:]:
            db.execute('insert into message_group (chat_id, message_id, group_id) values (?, ?, ?)', [chat_id, i['message_id'], messages[0]['message_id']])
        return messages[0]


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


@dataclasses.dataclass(frozen=True)
class BaseModel:
    name: str
    short_name: str
    reasoning_options: list[str]
    default_reasoning: str | None

    async def complete(self, model_short_name, model, history, file, text, safety_identifier):
        raise NotImplementedError

    def translate_reasoning(self, v):
        return {
            'dynamic': '自动',
            'none': '无',
            'minimal': '最低',
            'low': '低',
            'medium': '中',
            'high': '高',
            'xhigh': '极高',
        }[v]

    def display_reasoning(self, model):
        if 'reasoning' in model:
            return f'{self.translate_reasoning(model["reasoning"])}（{model["reasoning"]}）'
        if self.default_reasoning is None:
            return '不指定'
        return f'不指定，默认{self.translate_reasoning(self.default_reasoning)}（{self.default_reasoning}）'


@dataclasses.dataclass(frozen=True)
class BaseModelGPT5(BaseModel):
    async def complete(self, model_short_name, model, history, file, text, safety_identifier):
        if 'system' in model:
            assert history is None
            if model['system']:
                history = [{'role': 'system', 'content': [{'type': 'input_text', 'text': model['system']}]}]
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
            'model': self.name,
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

        text = f'[{model_short_name}]'
        search_text = ''
        for output in response['output']:
            match output['type']:
                case 'reasoning':
                    history.append(output)
                case 'web_search_call':
                    text += '[网页搜索]'
                case 'code_interpreter_call':
                    text += '[运行代码]'
                case 'message':
                    history.append(output)
                    for content in output['content']:
                        match content['type']:
                            case 'output_text':
                                text += content['text']
                                search_text += content['text']
                            case type:
                                text += f'[未知内容类型：{type}]'
                case type:
                    text += f'[未知输出类型：{type}]'
        return history, text, search_text


@dataclasses.dataclass(frozen=True)
class BaseModelGemini(BaseModel):
    def thinking_config(self, model):
        raise NotImplementedError

    async def complete(self, model_short_name, model, history, file, text, safety_identifier):
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
            kwargs['generationConfig'] = {'thinkingConfig': self.thinking_config(model)}
        if 'tools' in model:
            kwargs['tools'] = []
            for i in model['tools']:
                match i:
                    case 's':
                        kwargs['tools'].append({'googleSearch': {}})
                    case 'w':
                        kwargs['tools'].append({'urlContext': {}})
                    case 'c':
                        kwargs['tools'].append({'codeExecution': {}})
                    case _:
                        raise ValueError(f'Unknown tool option: {i}')
        if model.get('system', ''):
            kwargs['systemInstruction'] = {'parts': [{'text': model['system']}]}

        async with session.post(
            f'https://generativelanguage.googleapis.com/v1beta/models/{self.name}:generateContent',
            headers={'x-goog-api-key': gemini_api_key},
            json=kwargs,
            timeout=aiohttp.ClientTimeout(total=3600),
        ) as response:
            if response.status != 200:
                raise RuntimeError(f'Gemini API error: {response.status} {await response.text()}')
            response = await response.json()
        if 'candidates' not in response or len(response['candidates']) != 1:
            raise RuntimeError('Gemini API error: no candidates')

        text = f'[{model_short_name}]'
        search_text = ''
        content = response['candidates'][0]['content']
        for part in content['parts']:
            match part:
                case {'executableCode': _}:
                    text += '[运行代码]'
                case {'codeExecutionResult': _}:
                    pass
                case {'text': s}:
                    text += s
                    search_text += s
                case _:
                    if len(part) == 1:
                        text += f'[未知内容类型：{next(iter(part))}]'
                    else:
                        text += f'[未知内容类型：{part}]'
        history.append(content)
        return history, text, search_text


@dataclasses.dataclass(frozen=True)
class BaseModelGemini3(BaseModelGemini):
    def thinking_config(self, model):
        return {'thinkingLevel': model['reasoning']}


@dataclasses.dataclass(frozen=True)
class BaseModelGemini25(BaseModelGemini):
    reasoning_options: list[str] = dataclasses.field(init=False)
    default_reasoning: str | None = dataclasses.field(init=False)
    thinking_zero: bool
    thinking_min: int
    thinking_max: int
    default_thinking: int

    def __post_init__(self):
        if self.thinking_zero:
            object.__setattr__(self, 'reasoning_options', ['dynamic', 'none', 'minimal', 'low', 'medium', 'high'])
        else:
            object.__setattr__(self, 'reasoning_options', ['dynamic', 'minimal', 'low', 'medium', 'high'])
        match self.default_thinking:
            case -1:
                object.__setattr__(self, 'default_reasoning', 'dynamic')
            case 0:
                object.__setattr__(self, 'default_reasoning', 'none')
            case _:
                raise ValueError('Invalid default_thinking')

    def clip_budget(self, budget):
        if budget == -1:
            return -1
        if budget == 0 and self.thinking_zero:
            return 0
        return max(self.thinking_min, min(self.thinking_max, budget))

    def thinking_config(self, model):
        if isinstance(model['reasoning'], int):
            budget = model['reasoning']
        else:
            budget = self.clip_budget({
                'dynamic': -1,
                'none': 0,
                'minimal': 128,
                'low': 1024,
                'medium': 8192,
                'high': 32768,
            }[model['reasoning']])
        return {'thinkingBudget': budget}


base_models = [
    BaseModelGPT5('gpt-5.2', '52', ['none', 'low', 'medium', 'high', 'xhigh'], 'none'),
    BaseModelGPT5('gpt-5.1', '51', ['none', 'low', 'medium', 'high'], 'none'),
    BaseModelGPT5('gpt-5', '5', ['minimal', 'low', 'medium', 'high'], 'medium'),
    BaseModelGPT5('gpt-5-mini', '5m', ['minimal', 'low', 'medium', 'high'], 'medium'),
    BaseModelGPT5('gpt-5-nano', '5n', ['minimal', 'low', 'medium', 'high'], 'medium'),
    BaseModelGemini3('gemini-3-pro-preview', '3p', ['low', 'high'], 'high'),
    BaseModelGemini3('gemini-3-flash-preview', '3f', ['minimal', 'low', 'medium', 'high'], 'high'),
    BaseModelGemini25('gemini-2.5-pro', '25p', False, 128, 32768, -1),
    BaseModelGemini25('gemini-2.5-flash', '25f', True, 0, 24576, -1),
    BaseModelGemini25('gemini-2.5-flash-lite', '25l', True, 512, 24576, 0),
]
base_model_buttons = [
    ['52', '5m', '5n'],
    ['3p', '3f'],
]


def parse_base_model(name):
    for i in base_models:
        if i.short_name == name or i.name == name:
            return i
    if not name:
        raise ValueError('Must specify a model')
    if not re.fullmatch(r'[\w.-]+', name, re.ASCII):
        raise ValueError(f'Invalid model name: {name}')
    if name.startswith('gpt-'):
        return BaseModelGPT5(name, name, ['none', 'minimal', 'low', 'medium', 'high', 'xhigh'], None)
    if name.startswith('gemini-'):
        return BaseModelGemini3(name, name, ['minimal', 'low', 'medium', 'high'], None)
    raise ValueError(f'Model name must start with gpt- or gemini-: {name}')


def parse_prefix(prefix):
    models = []
    for part in prefix.split(','):
        part = part.strip()
        if not part:
            raise ValueError('Empty model specification')

        s, *args = part.split('+')
        s = s.strip()
        model = {'model': parse_base_model(s).name}

        model_type = model['model'].split('-', 1)[0]
        for arg in args:
            match arg[0], model_type:
                case 'r', _:
                    s = arg[1:].strip()
                    if not s:
                        raise ValueError('Missing reasoning effort')
                    if re.fullmatch(r'\d+', s, re.ASCII):
                        model['reasoning'] = int(s)
                    else:
                        for i in ['low', 'medium', 'high', 'minimal', 'none', 'dynamic', 'xhigh']:
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
                case 't', 'gpt':
                    s = arg[1:].strip()
                    if len(set(s)) != len(s):
                        raise ValueError(f'Duplicate tool options: {arg[1:]}')
                    if set(s) - {'s', 'c'}:
                        raise ValueError(f'Unknown tool options: {arg[1:]}')
                    if s:
                        model['tools'] = s
                case 't', 'gemini':
                    s = arg[1:].strip()
                    if len(set(s)) != len(s):
                        raise ValueError(f'Duplicate tool options: {arg[1:]}')
                    if set(s) - {'s', 'w', 'c'}:
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


def format_prefix(models, omit_system=False):
    parts = []
    for model in models:
        s = parse_base_model(model['model']).short_name
        if 'reasoning' in model:
            s += '+r'
            if isinstance(model['reasoning'], int):
                s += str(model['reasoning'])
            else:
                s += {
                    'low': 'l',
                    'medium': 'm',
                    'high': 'h',
                    'minimal': 'min',
                    'none': 'n',
                    'dynamic': 'd',
                    'xhigh': 'x',
                }[model['reasoning']]
        if 'verbosity' in model:
            s += '+v' + {'low': 'l', 'medium': 'm', 'high': 'h'}[model['verbosity']]
        if 'tools' in model:
            s += '+t' + model['tools']
        if 'system' in model:
            if not omit_system:
                if '+' in model['system'] or ',' in model['system'] or '$' in model['system']:
                    return None
                s += '+s' + model['system']
        parts.append(s)
    return ','.join(parts)


async def render_select_model_state(state, chat_id, message_id=None):
    kwargs = {
        'chat_id': chat_id,
        'parse_mode': 'HTML',
        'link_preview_options': {'is_disabled': True},
    }
    match state:
        case {'step': 'model'}:
            kwargs['text'] = '基础模型：请选择'
            l = []
            for row in base_model_buttons:
                button_row = []
                for i in row:
                    base_model = parse_base_model(i)
                    button_row.append({'text': base_model.name, 'callback_data': f'm/{base_model.name}'})
                l.append(button_row)
            l.append([{'text': '其他（不一定兼容）', 'callback_data': 'm/_other'}])
            kwargs['reply_markup'] = {'inline_keyboard': l}
        case {'step': 'model-input'}:
            kwargs['text'] = state.get('error', '请回复 OpenAI 或 Gemini 模型名称')
            kwargs['reply_markup'] = {'force_reply': True, 'input_field_placeholder': 'gpt-... / gemini-...'}
        case {'step': 'system-input'}:
            kwargs['text'] = state.get('error', '请回复系统提示。如果超过 4096 字符，必须以文本文件形式发送')
            kwargs['reply_markup'] = {'force_reply': True, 'input_field_placeholder': '你是一个...'}
        case {'step': 'ready' | 'used' | 'invalid'}:
            lines = ['基础模型：' + escape_html(state['model'])]
            base_model = parse_base_model(state['model'])
            lines.append('推理努力：' + base_model.display_reasoning(state))
            model_type = state['model'].split('-', 1)[0]
            if model_type == 'gpt':
                lines.append('输出长度：' + state.get('verbosity', '不指定（默认 medium）'))
            lines.append('在线搜索：' + ('开' if 's' in state.get('tools', '') else '关'))
            if model_type == 'gemini':
                lines.append('查看网页：' + ('开' if 'w' in state.get('tools', '') else '关'))
            lines.append('运行代码：' + ('开' if 'c' in state.get('tools', '') else '关'))
            if 'system' in state:
                text = state['system']
                if len(text) > 2002:
                    text = text[:2000] + '……'
                lines.append('系统提示：')
                lines.append('<blockquote expandable>' + escape_html(text) + '</blockquote>')
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
            l = [{'text': '推理：', 'callback_data': 'r/'}]
            for i in base_model.reasoning_options:
                l.append({'text': base_model.translate_reasoning(i), 'callback_data': f'r/{i}'})
            keyboard.append(l)
            if model_type == 'gpt':
                keyboard.append([
                    {'text': '长度：', 'callback_data': 'v/'},
                    {'text': 'low', 'callback_data': 'v/low'},
                    {'text': 'medium', 'callback_data': 'v/medium'},
                    {'text': 'high', 'callback_data': 'v/high'},
                ])
                keyboard.append([
                    {
                        True: {'text': '在线搜索关', 'callback_data': 'tn/s'},
                        False: {'text': '在线搜索开', 'callback_data': 'ty/s'},
                    }['s' in state.get('tools', '')],
                    {
                        True: {'text': '运行代码关', 'callback_data': 'tn/c'},
                        False: {'text': '运行代码开', 'callback_data': 'ty/c'},
                    }['c' in state.get('tools', '')],
                ])
            elif model_type == 'gemini':
                keyboard.append([
                    {
                        True: {'text': '在线搜索关', 'callback_data': 'tn/s'},
                        False: {'text': '在线搜索开', 'callback_data': 'ty/s'},
                    }['s' in state.get('tools', '')],
                    {
                        True: {'text': '查看网页关', 'callback_data': 'tn/w'},
                        False: {'text': '查看网页开', 'callback_data': 'ty/w'},
                    }['w' in state.get('tools', '')],
                    {
                        True: {'text': '运行代码关', 'callback_data': 'tn/c'},
                        False: {'text': '运行代码开', 'callback_data': 'ty/c'},
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
    base_model = parse_base_model(state['model'])
    if 'reasoning' in state and state['reasoning'] not in base_model.reasoning_options:
        del state['reasoning']
    model_type = state['model'].split('-', 1)[0]
    if model_type == 'gpt':
        if 'w' in state.get('tools', ''):
            state['tools'] = state['tools'].replace('w', '')
            if not state['tools']:
                del state['tools']
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
    # if 'photo' in message:
    #     state['error'] = '不支持使用图片作为系统提示，请回复文本消息或文本文件作为系统提示'
    #     await render_select_model_state(state, chat_id)
    #     return
    # if 'document' in message:
    #     if message.get('text', '').strip():
    #         state['error'] = '发送文本文件时请不要在消息中添加其他内容，请回复文本文件作为系统提示'
    #         await telegram_send_text(chat_id, '请回复你发送的文档。', reply_to_message_id=message_id)
    # text = message['text']
    match state['step']:
        case 'model-input':
            if 'photo' in message or 'document' in message:
                state['error'] = '请回复文本消息作为模型名称'
                await render_select_model_state(state, chat_id)
                return
            model = message['text'].strip().lower()
            if not model.startswith(('gpt-', 'gemini-')):
                state['error'] = '模型名称必须以 gpt- 或 gemini- 开头，请重新输入'
                await render_select_model_state(state, chat_id)
                return
            if not re.fullmatch(r'(?a)[\w.-]+', model):
                state['error'] = f'模型名称只能包含字母、数字、点号、下划线和连字符，请重新输入'
                await render_select_model_state(state, chat_id)
                return
            state['model'] = model
            state['step'] = 'ready'
            state.pop('error', None)
            select_model_after_change_model(state)
            select_model_check_invalid(state)
            await render_select_model_state(state, chat_id)
        case 'system-input':
            if 'photo' in message:
                state['error'] = '不支持使用图片作为系统提示，请回复文本消息或文本文件作为系统提示'
                await render_select_model_state(state, chat_id)
                return
            elif 'document' in message:
                if message.get('text', '').strip():
                    state['error'] = '不支持发送带额外文本消息的文件作为系统提示，请回复文本消息或文本文件作为系统提示'
                    await render_select_model_state(state, chat_id)
                    return
                if not message['document'].get('mime_type', '').startswith('text/'):
                    state['error'] = '不支持使用非文本文件作为系统提示，请回复文本消息或文本文件作为系统提示'
                    await render_select_model_state(state, chat_id)
                    return
                file_id = message['document']['file_id']
                text = (await telegram_get_file(file_id)).decode(errors='replace')
            else:
                text = message.get('text', '')
            state['system'] = text.strip()
            state['step'] = 'ready'
            state.pop('error', None)
            await render_select_model_state(state, chat_id)
        case _:
            raise ValueError('Invalid reply message')


# async def handle_search_message(message):
#     chat_id = message['chat']['id']
#     _, *args = message.get('text', '').split()
#     if not args:
#         await telegram_send_text(chat_id, '用法：/search 关键词1 关键词2 ...')
#         return
#     state = {'type': 'search', 'args': args, 'page': }
#     db.execute('insert into message (chat_id, message_id, data) values (?, ?, ?)', [chat_id, message['message_id'], serialize_fast({'type': 'search', 'args': args})])
#     like_part = ' and '.join(['text like ?'] * len(args))
#     db.execute('select * from search_data where chat_id = ? and ' + like_part + ' order by message_id desc limit 10


# async def handle_search_callback_query(callback_query):
#     pass


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
        # if message.get('text', '').startswith('/search'):
        #     await handle_search_message(message)
        #     return

        message_id = message['message_id']
        if 'media_group_id' in message:
            await telegram_send_text(chat_id, '只支持每次发送一张图片或一个文件。', reply_to_message_id=message_id)
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

        reply_file = None
        history = None
        models = None
        search_data_parent = None
        if 'reply_to_message' in message:
            reply_to_message_id = message['reply_to_message']['message_id']
            if row := db.execute('select group_id from message_group where chat_id = ? and message_id = ?', [chat_id, reply_to_message_id]).fetchone():
                reply_to_message_id = row[0]
            if (row := db.execute('select data from message where chat_id = ? and message_id = ?', [message['chat']['id'], reply_to_message_id]).fetchone()) is None:
                await telegram_send_text(chat_id, '只支持回复 AI 发送的消息，或者你发送的图片或文件。', reply_to_message_id=message_id)
                return
            match deserialize(row[0]):
                case {'type': 'history', 'history': history, 'models': models}:
                    if row := db.execute('select 1 from search_data where chat_id = ? and message_id = ?', [chat_id, reply_to_message_id]).fetchone():
                        search_data_parent = reply_to_message_id
                case {'type': 'file'} as reply_file:
                    pass
                case {'type': 'select_model', 'step': 'ready' | 'used'} as state:
                    if state['step'] == 'ready':
                        state['step'] = 'used'
                        db.execute('insert into message (chat_id, message_id, data) values (?, ?, ?) on conflict do update set data = excluded.data', [chat_id, reply_to_message_id, serialize_fast(state)])
                    models = [{k: v for k, v in state.items() if k in {'model', 'reasoning', 'verbosity', 'tools', 'system'}}]
                case {'type': 'select_model', 'step': 'model-input' | 'system-input'} as state:
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

        db.execute('begin immediate')
        with db:
            db.execute('insert into search_data (chat_id, message_id, text, parent, time) values (?, ?, ?, ?, ?)', [chat_id, message_id, text or '', search_data_parent, message['date']])
            if search_data_parent is not None:
                parent_children, = db.execute('select children from search_data where chat_id = ? and message_id = ?', [chat_id, search_data_parent]).fetchone()
                if parent_children is None:
                    parent_children = []
                else:
                    parent_children = deserialize(parent_children)
                parent_children.append(message_id)
                db.execute('update search_data set children = ? where chat_id = ? and message_id = ?', [serialize_fast(parent_children), chat_id, search_data_parent])

        [safety_identifier_salt] = db.execute('select value from config where key = ?', ['safety_identifier_salt']).fetchone()
        if len(safety_identifier_salt) < 16:
            raise ValueError('safety_identifier_salt too short')
        safety_identifier = hashlib.sha256(f'{safety_identifier_salt},{from_id}'.encode()).hexdigest()
        for model in models:
            model_short_name = format_prefix([model], omit_system=True)
            asyncio.create_task(complete_and_reply(chat_id, message_id, model_short_name, model, history, file, text, safety_identifier, strict_privacy))

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


async def complete_and_reply(chat_id, message_id, model_short_name, model, history, file, text, safety_identifier, strict_privacy):
    try:
        base_model = parse_base_model(model['model'])
        history, text, search_text = await base_model.complete(model_short_name, model, history, file, text, safety_identifier)

        result = await telegram_send_text(chat_id, text, reply_to_message_id=message_id)
        db.execute('insert into message (chat_id, message_id, data) values (?, ?, ?)', [chat_id, result['message_id'], serialize_fast({
            'type': 'history',
            'history': history,
            'models': [model],
        })])

        db.execute('begin immediate')
        with db:
            db.execute('insert into search_data (chat_id, message_id, text, parent, time) values (?, ?, ?, ?, ?)', [chat_id, result['message_id'], search_text, message_id, result['date']])
            parent_children, = db.execute('select children from search_data where chat_id = ? and message_id = ?', [chat_id, message_id]).fetchone()
            if parent_children is None:
                parent_children = []
            else:
                parent_children = deserialize(parent_children)
            parent_children.append(result['message_id'])
            db.execute('update search_data set children = ? where chat_id = ? and message_id = ?', [serialize_fast(parent_children), chat_id, message_id])

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
