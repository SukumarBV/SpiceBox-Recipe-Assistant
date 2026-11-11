"""SpiceBox visual theme (CSS). System-font fallbacks are provided if web fonts cannot load."""
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');
:root{--o:#FC8019;--od:#E86F08;--tx:#29221D;--tx2:#766C64;--bg:#FFF9F3;--card:#fff;--bd:#F0E3D7;--soft:#FFF0E1;--ok:#26834A;--warn:#B7791F;--err:#C0392B;}
html,body,[class*="css"],.stApp{font-family:'Inter',system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;color:var(--tx);}
.stApp{background:var(--bg);}
header[data-testid="stHeader"],footer,#MainMenu,[data-testid="stToolbar"],[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none!important;}
.block-container{max-width:1160px;padding:.6rem 1.2rem 6rem;}
h1,h2,h3{font-family:'Fraunces',Georgia,'Times New Roman',serif;color:var(--tx);letter-spacing:-.01em;}
.sb-header{display:flex;align-items:center;gap:18px;padding:10px 0 12px;border-bottom:1px solid var(--bd);flex-wrap:wrap;}
.sb-brand{display:flex;align-items:center;gap:10px;font-family:'Fraunces',Georgia,serif;font-weight:700;font-size:1.35rem;}
.sb-logo{width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,var(--o),var(--od));display:grid;place-items:center;box-shadow:0 4px 12px rgba(252,128,25,.35);}
.sb-brand small{display:block;font-family:'Inter',sans-serif;font-weight:500;font-size:.72rem;color:var(--tx2);margin-top:-2px;}
.sb-nav{display:flex;gap:6px;flex-wrap:wrap;}
.sb-nav a{color:var(--tx2);text-decoration:none;font-weight:600;font-size:.9rem;padding:7px 12px;border-radius:999px;}
.sb-nav a:hover{background:var(--soft);color:var(--od);}
.sb-status{margin-left:auto;display:inline-flex;align-items:center;gap:8px;font-size:.8rem;font-weight:600;color:var(--tx2);background:#fff;border:1px solid var(--bd);padding:6px 12px;border-radius:999px;}
.sb-dot{width:9px;height:9px;border-radius:50%;background:#bbb;display:inline-block}.sb-dot.on{background:var(--ok)}.sb-dot.pause{background:var(--warn)}
.sb-hero{padding:34px 0 8px;display:grid;grid-template-columns:1.5fr 1fr;gap:28px;align-items:center;}
.sb-eyebrow{color:var(--od);font-weight:700;font-size:.78rem;letter-spacing:.14em;text-transform:uppercase;}
.sb-hero h1{font-size:clamp(2.1rem,5.2vw,3.7rem);line-height:1.04;margin:.35rem 0 .7rem;font-weight:700;}
.sb-hero h1 em{color:var(--o);font-style:normal;}
.sb-hero p{color:var(--tx2);font-size:1.06rem;line-height:1.55;max-width:34rem;margin:0;}
.sb-plate{background:linear-gradient(150deg,#FFE6CC,#FFD2A3);border-radius:28px;padding:26px;border:1px solid var(--bd);box-shadow:0 18px 40px rgba(232,111,8,.12);}
.sb-plate h4{margin:0 0 10px;font-family:'Inter',sans-serif;font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;color:var(--od);}
.sb-stat{display:flex;justify-content:space-between;align-items:baseline;padding:9px 0;border-top:1px dashed rgba(41,34,29,.18);}
.sb-stat b{font-family:'Fraunces',Georgia,serif;font-size:1.5rem}.sb-stat span{color:var(--tx2);font-size:.85rem;font-weight:500}
@media(max-width:820px){.sb-hero{grid-template-columns:1fr}.sb-status{margin-left:0}}
/* inputs */
div[data-testid="stTextInput"] input,div[data-testid="stNumberInput"] input,div[data-testid="stSelectbox"] div[data-baseweb="select"]>div{border-radius:14px!important;border:1.5px solid var(--bd)!important;background:#fff!important;}
div[data-testid="stTextInput"] input:focus{border-color:var(--o)!important;box-shadow:0 0 0 3px rgba(252,128,25,.18)!important;}
div[data-testid="stForm"]{border:none!important;padding:0!important;background:transparent!important;}
.stButton>button,div[data-testid="stFormSubmitButton"]>button{border-radius:999px;font-weight:600;border:1.5px solid var(--bd);background:#fff;color:var(--tx);padding:.45rem 1rem;transition:all .15s;}
.stButton>button:hover,div[data-testid="stFormSubmitButton"]>button:hover{border-color:var(--o);color:var(--od);transform:translateY(-1px);box-shadow:0 6px 14px rgba(252,128,25,.15);}
.stButton>button[kind="primary"],div[data-testid="stFormSubmitButton"]>button[kind="primaryFormSubmit"],div[data-testid="stFormSubmitButton"]>button[kind="primary"]{background:var(--o);border-color:var(--o);color:#fff;}
.stButton>button[kind="primary"]:hover,div[data-testid="stFormSubmitButton"]>button[kind="primaryFormSubmit"]:hover{background:var(--od);color:#fff;}
button[data-baseweb="tab"]{font-weight:600;color:var(--tx2);}button[data-baseweb="tab"][aria-selected="true"]{color:var(--od);}
div[data-baseweb="tab-highlight"]{background:var(--o)!important;}
/* cards */
div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--card);border:1px solid var(--bd)!important;border-radius:18px!important;box-shadow:0 2px 10px rgba(41,34,29,.04);transition:box-shadow .15s,transform .15s;}
div[data-testid="stVerticalBlockBorderWrapper"]:hover{box-shadow:0 10px 26px rgba(41,34,29,.08);}
.rc-top{display:flex;gap:12px;align-items:flex-start}
.rc-badge{flex:0 0 42px;height:42px;border-radius:13px;background:linear-gradient(135deg,var(--soft),#FFD9B5);display:grid;place-items:center;font-family:'Fraunces',Georgia,serif;font-weight:700;color:var(--od);font-size:1.2rem}
.rc-title{font-family:'Fraunces',Georgia,serif;font-weight:700;font-size:1.12rem;line-height:1.25}
.rc-id{font-size:.74rem;color:var(--tx2);font-family:ui-monospace,Consolas,monospace}
.rc-score{margin-left:auto;text-align:right;white-space:nowrap}.rc-score b{display:block;font-size:1.35rem;color:var(--od);font-family:'Fraunces',Georgia,serif}.rc-score span{font-size:.7rem;color:var(--tx2);text-transform:uppercase;letter-spacing:.08em}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 2px}
.chip{display:inline-block;font-size:.78rem;font-weight:600;padding:3px 10px;border-radius:999px;background:var(--soft);color:#8a4a10;border:1px solid #f7dcc2}
.chip.ok{background:#E8F5EC;color:var(--ok);border-color:#cfe8d7}.chip.warn{background:#FFF4DC;color:var(--warn);border-color:#f3e1b4}.chip.err{background:#FBE9E7;color:var(--err);border-color:#f2cbc6}.chip.mute{background:#F4EFEA;color:var(--tx2);border-color:#e8dfd6}
.lbl{font-size:.72rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--tx2);margin:8px 0 0}
.note{font-size:.8rem;color:var(--tx2);margin:6px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:6px 0 10px}
.tile{background:#fff;border:1px solid var(--bd);border-radius:14px;padding:12px 14px}.tile b{display:block;font-size:1.5rem;font-family:'Fraunces',Georgia,serif;color:var(--od)}.tile span{font-size:.75rem;color:var(--tx2);font-weight:600;text-transform:uppercase;letter-spacing:.06em}
table.sb-table{width:100%;border-collapse:collapse;font-size:.88rem;background:#fff;border:1px solid var(--bd);border-radius:12px;overflow:hidden}
table.sb-table th{background:var(--soft);text-align:left;padding:8px 10px;font-size:.74rem;text-transform:uppercase;letter-spacing:.06em;color:#8a4a10}
table.sb-table td{padding:8px 10px;border-top:1px solid var(--bd);vertical-align:top}
.tbl-wrap{overflow-x:auto}
.ctx{display:flex;flex-wrap:wrap;gap:8px;align-items:center;background:#fff;border:1px solid var(--bd);border-radius:16px;padding:10px 14px;margin:10px 0}
.ctx .k{font-size:.72rem;font-weight:700;color:var(--tx2);text-transform:uppercase;letter-spacing:.08em;margin-right:4px}
.banner{border-radius:14px;padding:10px 14px;font-size:.88rem;margin:8px 0;border:1px solid}
.banner.ok{background:#E8F5EC;border-color:#cfe8d7;color:#1c5e35}.banner.warn{background:#FFF4DC;border-color:#f3e1b4;color:#80560f}.banner.err{background:#FBE9E7;border-color:#f2cbc6;color:#8f2b20}.banner.info{background:var(--soft);border-color:#f7dcc2;color:#7a4510}
.sec{font-family:'Fraunces',Georgia,serif;font-size:1.5rem;margin:26px 0 6px}
div[data-testid="stChatMessage"]{background:transparent;}
@media(max-width:640px){.rc-top{flex-wrap:wrap}.rc-score{margin-left:0;text-align:left}.block-container{padding:.4rem .7rem 6rem}}
</style>
"""
