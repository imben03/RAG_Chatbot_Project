"""
Run this script from your project folder:
    python patch_app_icon.py

It will:
1. Patch app.py to use chatbot_icon.png instead of the graduation cap emoji
2. chatbot_icon.png must be in the same folder as app.py
"""
import re

with open('app.py', 'r', encoding='utf-8') as f:
    code = f.read()

# --- CHANGE 1: Add base64 import (if not already present) ---
if 'import base64' not in code:
    code = code.replace(
        'from dotenv import load_dotenv',
        'from dotenv import load_dotenv\nimport base64'
    )

# --- CHANGE 2: Add icon loader after load_dotenv() ---
icon_loader = '''
# Load chatbot icon as base64 for embedding in HTML
def load_icon_b64(path='chatbot_icon.png'):
    try:
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ''

ICON_B64 = load_icon_b64()
ICON_DATA_URI = f'data:image/png;base64,{ICON_B64}' if ICON_B64 else ''
'''

if 'ICON_DATA_URI' not in code:
    code = code.replace(
        'load_dotenv()\n',
        'load_dotenv()\n' + icon_loader
    )

# --- CHANGE 3: page_icon ---
code = code.replace("page_icon='🎓'", "page_icon='chatbot_icon.png'")

# --- CHANGE 4: Sidebar brand icon ---
# Old: <span class="brand-icon">🎓</span>
# New: <img> tag with base64 data URI
old_brand = '<span class="brand-icon">🎓</span>'
new_brand = '<img src="{ICON_DATA_URI}" class="brand-icon" style="width:38px;height:38px;border-radius:10px;vertical-align:middle;"/>'
code = code.replace(old_brand, new_brand)

# Make the sidebar brand st.markdown an f-string so {ICON_DATA_URI} resolves
code = code.replace(
    '    st.markdown("""\n    <div class="sidebar-brand">',
    '    st.markdown(f"""\n    <div class="sidebar-brand">'
)

# --- CHANGE 5: Main header crest ---
old_crest = '<span class="crest">🎓</span>'
new_crest = '<img src="{ICON_DATA_URI}" class="crest" style="width:52px;height:52px;border-radius:14px;vertical-align:middle;margin-right:12px;"/>'
code = code.replace(old_crest, new_crest)

# Make the header st.markdown an f-string
code = code.replace(
    'st.markdown("""\n<div class="mum-header">',
    'st.markdown(f"""\n<div class="mum-header">'
)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(code)

print('✔ app.py patched successfully!')
print('  - page_icon changed to chatbot_icon.png')
print('  - Sidebar brand icon changed to embedded chatbot image')
print('  - Header crest changed to embedded chatbot image')
print('')
print('Make sure chatbot_icon.png is in the same folder as app.py.')