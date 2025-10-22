import os, json, datetime
from .utils import ensure_dir, atomic_write_json, atomic_write_text, append_jsonl

def list_turn_dirs(session_dir): 
    return sorted([d for d in os.listdir(session_dir) if d.startswith("turn-")])

def next_turn_index(session_dir):
    ds=list_turn_dirs(session_dir)
    if not ds: return 1
    try: return max(int(d.split("-")[-1]) for d in ds)+1
    except: return len(ds)+1

class ChatSession:
    def __init__(self, session_root, session_name=None, naming="timestamp", cfg=None):
        ensure_dir(session_root)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if session_name and naming=="name_and_timestamp": folder=f"{session_name}-{ts}"
        elif session_name and naming=="name_only": folder=session_name
        else: folder=f"session-{ts}"
        self.session_dir=os.path.join(session_root,folder); ensure_dir(self.session_dir)
        self.summary_path=os.path.join(self.session_dir,"summary.txt")
        self.transcript_path=os.path.join(self.session_dir,"transcript.jsonl")
        self.focus_path=os.path.join(self.session_dir,"focus.json")
        self.entities_path=os.path.join(self.session_dir,"entities.json")
        self.turn_idx=next_turn_index(self.session_dir)

    def new_turn_dir(self):
        d=os.path.join(self.session_dir,f"turn-{self.turn_idx:04d}"); ensure_dir(d); self.turn_idx+=1; return d

    def read_last_turn_summary(self):
        ds=list_turn_dirs(self.session_dir)
        if not ds: return ""
        p=os.path.join(self.session_dir,ds[-1],"turn_summary.txt")
        return open(p,"r",encoding="utf-8").read() if os.path.exists(p) else ""

    def append_transcript(self, rec):
        rec["ts"]=datetime.datetime.now().isoformat()
        append_jsonl(self.transcript_path,[rec])

    def read_chat_summary(self):
        return open(self.summary_path,"r",encoding="utf-8").read() if os.path.exists(self.summary_path) else ""

    def write_chat_summary(self, txt):
        atomic_write_text(self.summary_path, txt or "")

    def get_focus(self):
        if os.path.exists(self.focus_path):
            try: return json.load(open(self.focus_path,"r",encoding="utf-8")).get("last_focus","")
            except: return ""
        return ""

    def set_focus(self,name):
        if name: atomic_write_json(self.focus_path, {"last_focus":name,"updated_at":datetime.datetime.now().isoformat()})

    def get_entities(self):
        if os.path.exists(self.entities_path):
            try: return json.load(open(self.entities_path,"r",encoding="utf-8")).get("stack",[])
            except: return []
        return []

    def push_entities(self, ents, max_keep=64):
        stack=self.get_entities()
        for e in ents:
            if e and e not in stack: stack.append(e)
        stack=stack[-max_keep:]
        atomic_write_json(self.entities_path, {"stack":stack,"updated_at":datetime.datetime.now().isoformat()})
