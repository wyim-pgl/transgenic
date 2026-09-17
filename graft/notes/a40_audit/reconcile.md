# notes/a40_audit/reconcile.py

- load · function · L8-L10 — def load(path,name,text=None)
- signature · function · L13-L15 — def signature(d)
- read_rows · function · L16-L21 — def read_rows(con)
- Capture · class · L22-L35 — class Capture
- __init__ · method · L23-L23 — def __init__(self)
- sql · method · L24-L24 — def sql(self,*a,**kw)
- execute · method · L25-L30 — def execute(self,q,p=None)
- executemany · method · L31-L34 — def executemany(self,q,rows)
- fetchall · method · L35-L35 — def fetchall(self)
- replay · function · L37-L95 — def replay(sp,con,counterfactual=False)
- observe · function · L46-L74 — def observe(stage,v)
