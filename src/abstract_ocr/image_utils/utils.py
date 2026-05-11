from ..init_imports import *
def create_key_value(json_obj, key, value):
    json_obj[key] = json_obj.get(key, value) or value
    return json_obj

def getPercent(i):
    return divide_it(i, 100)

def getPercentage(num, i):
    percent = getPercent(i)
    percentage = multiply_it(num, percent)
    return percentage

def if_none_get_def(value, default):
    if value is None:
        value = default
    return value

def if_not_dir_return_None(directory):
    str_directory = str(directory)
    if os.path.isdir(str_directory):
        return str_directory
    return None

def determine_remove_text(text,remove_phrases=None):
    remove_phrases=remove_phrases or []
    found = False
    for remove_phrase in remove_phrases:
        if remove_phrase in text:
            found = True
            break
    if found == False:
        return text

def generate_file_id(path: str, max_length: int = 50) -> str:
    base = os.path.splitext(os.path.basename(path))[0]
    base = unicodedata.normalize('NFKD', base).encode('ascii', 'ignore').decode('ascii')
    base = base.lower()
    base = re.sub(r'[^a-z0-9]+', '-', base).strip('-')
    base = re.sub(r'-{2,}', '-', base)
    if len(base) > max_length:
        h = hashlib.sha1(base.encode()).hexdigest()[:8]
        base = base[: max_length - len(h) - 1].rstrip('-') + '-' + h
    return base
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s:.,-]', '', text)
    text = text.strip()
    return text
def get_frame_number(obj):
    if isinstance(obj,dict):
        obj = obj.get('frame')
    frame = '.'.join(obj.split('.')[:-1])
    return int(frame.split('_')[-1])
def sort_frames(frames=None,directory=None):
    if frames in [None,[]] and directory and os.path.isdir(directory):
        frames = get_all_file_types(types=['image'],directory=directory)
    frames = frames or []
    frames = sorted(
        frames,
        key=lambda x: get_frame_number(x) 
    )
    return frames
