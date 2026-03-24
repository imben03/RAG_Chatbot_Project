# maintain.py - Database Maintenance Tool
# CST3990 - Middlesex University Mauritius RAG Chatbot
#
# Usage:
#   python maintain.py --status              Show full database health report
#   python maintain.py --clean               Remove chunks for deleted/missing files
#   python maintain.py --remove filename.pdf Remove one specific document
#   python maintain.py --backup              Backup database to a zip file
#   python maintain.py --rebuild             Wipe and rebuild everything from scratch
#   python maintain.py --reset-log           Rebuild index_log.json from ChromaDB

#improting libraries
import os
import sys
import json
import shutil
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
import chromadb
from dotenv import load_dotenv

load_dotenv()

DB_PATH    = './chroma_store'
INDEX_LOG  = './index_log.json'
DOCS_DIR   = './docs'
BACKUP_DIR = './backups'
COLLECTION = 'mum_policy_docs'

# Colors for terminal output

GREEN  = '\033[92m'
YELLOW = '\033[93m'
RED    = '\033[91m'
CYAN   = '\033[96m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

def ok(msg):    print(f'  {GREEN}✔{RESET}  {msg}')
def warn(msg):  print(f'  {YELLOW}⚠{RESET}  {msg}')
def err(msg):   print(f'  {RED}✘{RESET}  {msg}')
def info(msg):  print(f'  {CYAN}ℹ{RESET}  {msg}')
def head(msg):  print(f'\n{BOLD}{msg}{RESET}')
def line():     print('─' * 52)

# Helper functions for loading/saving index log, accessing ChromaDB, and file operations

def load_index_log():
    if os.path.exists(INDEX_LOG):
        with open(INDEX_LOG, 'r') as f:
            return json.load(f)
    return {}

def save_index_log(log):
    with open(INDEX_LOG, 'w') as f:
        json.dump(log, f, indent=2)

def get_collection():
    chroma = chromadb.PersistentClient(path=DB_PATH)
    return chroma.get_or_create_collection(COLLECTION)

def get_docs_in_folder():
    supported = {'.pdf', '.docx', '.html', '.htm'}
    if not os.path.exists(DOCS_DIR):
        return []
    return [fp for fp in Path(DOCS_DIR).rglob('*') if fp.suffix.lower() in supported]

def file_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

def get_folder_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total

def format_size(bytes):
    if bytes < 1024:
        return f'{bytes} B'
    elif bytes < 1024 ** 2:
        return f'{bytes/1024:.1f} KB'
    else:
        return f'{bytes/1024**2:.1f} MB'

#Status report command: shows database status, indexed documents, and health checks

def cmd_status():
    head('DATABASE STATUS REPORT')
    line()

    # 1. ChromaDB
    head('1. ChromaDB')
    if not os.path.exists(DB_PATH):
        err('chroma_store/ folder not found. Run ingest.py first.')
        return

    try:
        col   = get_collection()
        count = col.count()
        size  = format_size(get_folder_size(DB_PATH))
        ok(f'Database found at {DB_PATH}')
        ok(f'Total chunks stored: {BOLD}{count}{RESET}')
        ok(f'Database size on disk: {BOLD}{size}{RESET}')

        # Sources in DB
        if count > 0:
            result  = col.get(include=['metadatas'])
            sources = {}
            for m in result['metadatas']:
                src = m.get('source', 'unknown')
                sources[src] = sources.get(src, 0) + 1
            print()
            info(f'{len(sources)} document(s) indexed in ChromaDB:')
            for src, n in sorted(sources.items()):
                print(f'      📄  {src} ({n} chunks)')
    except Exception as e:
        err(f'Could not read ChromaDB: {e}')
        return

    # 2. Index log
    head('2. index_log.json')
    log = load_index_log()
    if not log:
        warn('index_log.json is empty or missing.')
    else:
        ok(f'{len(log)} record(s) in index log')
        for name, meta in sorted(log.items()):
            dated = meta.get('indexed_at', 'unknown')[:10]
            chunks = meta.get('chunks', '?')
            print(f'      📑  {name} — {chunks} chunks — indexed {dated}')

    # 3. Docs folder
    head('3. docs/ Folder')
    doc_files = get_docs_in_folder()
    if not doc_files:
        warn('No supported documents found in docs/')
    else:
        ok(f'{len(doc_files)} document(s) in docs/ folder')
        for fp in sorted(doc_files):
            print(f'      📁  {fp.name}')

    # 4. Health checks
    head('4. Health Checks')
    issues = 0

    # Stale: in log but not in docs/
    doc_names = {fp.name for fp in doc_files}
    for name in list(log.keys()):
        if name not in doc_names:
            warn(f'STALE: "{name}" is in the index but file no longer exists in docs/')
            warn(f'       Run --clean to remove its chunks from the database.')
            issues += 1

    # Missing from log but in DB
    if count > 0:
        result  = col.get(include=['metadatas'])
        db_srcs = {m.get('source', '') for m in result['metadatas']}
        for src in db_srcs:
            if src and src not in log:
                warn(f'ORPHAN: "{src}" has chunks in DB but no entry in index_log.json')
                warn(f'        Run --reset-log to fix this.')
                issues += 1

    # Changed files (hash mismatch)
    for fp in doc_files:
        if fp.name in log:
            current_hash = file_hash(str(fp))
            if log[fp.name].get('hash') != current_hash:
                warn(f'CHANGED: "{fp.name}" has been modified since last index.')
                warn(f'         Run ingest.py to re-index it.')
                issues += 1

    if issues == 0:
        ok('All health checks passed. Database is clean.')

    line()
    print(f'\n  Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

#Clean command: removes chunks for files that are in the index log but no longer exist in the docs/ folder

def cmd_clean():
    head('CLEAN — Remove Chunks for Missing Files')
    line()

    log       = load_index_log()
    doc_files = {fp.name for fp in get_docs_in_folder()}
    col       = get_collection()

    stale = [name for name in log if name not in doc_files]

    if not stale:
        ok('Nothing to clean. All indexed files still exist in docs/')
        return

    warn(f'Found {len(stale)} stale document(s) to remove:')
    for name in stale:
        print(f'      - {name}')

    confirm = input('\n  Proceed with removal? (yes/no): ').strip().lower()
    if confirm != 'yes':
        info('Cancelled.')
        return

    for name in stale:
        try:
            result        = col.get(include=['metadatas'])
            ids_to_delete = [
                result['ids'][i]
                for i, m in enumerate(result['metadatas'])
                if m.get('source') == name
            ]
            if ids_to_delete:
                col.delete(ids=ids_to_delete)
                ok(f'Removed {len(ids_to_delete)} chunks for "{name}"')
            del log[name]
        except Exception as e:
            err(f'Failed to remove "{name}": {e}')

    save_index_log(log)
    ok('index_log.json updated.')
    ok(f'Database now contains {col.count()} chunks.')

#   Remove command: removes chunks for one specific document by filename

def cmd_remove(filename):
    head(f'REMOVE — "{filename}"')
    line()

    col = get_collection()
    log = load_index_log()

    # Check if it exists in DB
    result  = col.get(include=['metadatas'])
    db_srcs = [m.get('source') for m in result['metadatas']]

    if filename not in db_srcs and filename not in log:
        err(f'"{filename}" not found in database or index log.')
        return

    chunk_count = db_srcs.count(filename)
    warn(f'This will remove {chunk_count} chunk(s) for "{filename}".')
    confirm = input('  Proceed? (yes/no): ').strip().lower()
    if confirm != 'yes':
        info('Cancelled.')
        return

    ids_to_delete = [
        result['ids'][i]
        for i, m in enumerate(result['metadatas'])
        if m.get('source') == filename
    ]
    if ids_to_delete:
        col.delete(ids=ids_to_delete)
        ok(f'Removed {len(ids_to_delete)} chunks.')

    if filename in log:
        del log[filename]
        save_index_log(log)
        ok('Removed from index_log.json.')

    ok(f'Database now contains {col.count()} chunks.')

# Backup command: creates a zip file backup of the chroma_store/ folder and index_log.json with a timestamped name

def cmd_backup():
    head('BACKUP — Save Database to Zip')
    line()

    if not os.path.exists(DB_PATH):
        err('chroma_store/ not found. Nothing to back up.')
        return

    os.makedirs(BACKUP_DIR, exist_ok=True)
    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f'backup_{timestamp}'
    backup_path = os.path.join(BACKUP_DIR, backup_name)

    info('Creating backup...')

    # Zip chroma_store
    shutil.make_archive(backup_path, 'zip', '.', DB_PATH)

    # Also copy index_log.json into backup folder
    if os.path.exists(INDEX_LOG):
        log_backup = os.path.join(BACKUP_DIR, f'index_log_{timestamp}.json')
        shutil.copy2(INDEX_LOG, log_backup)
        ok(f'index_log.json saved to {log_backup}')

    final_zip  = backup_path + '.zip'
    zip_size   = format_size(os.path.getsize(final_zip))
    ok(f'Backup created: {final_zip}')
    ok(f'Backup size: {zip_size}')
    info('To restore: unzip the file and replace chroma_store/ with its contents.')

# Rebuild command: deletes the entire chroma_store/ folder and index_log.json,
#  then runs ingest.py to re-index everything from scratch

def cmd_rebuild():
    head('REBUILD — Wipe and Re-index Everything')
    line()

    warn('This will DELETE the entire database and index log, then re-index all docs.')
    warn('Make sure you have run --backup first if you want to keep a copy.')
    confirm = input('\n  Type "REBUILD" to confirm: ').strip()
    if confirm != 'REBUILD':
        info('Cancelled.')
        return

    # Remove chroma_store
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        ok('Deleted chroma_store/')

    # Remove index log
    if os.path.exists(INDEX_LOG):
        os.remove(INDEX_LOG)
        ok('Deleted index_log.json')

    info('Running ingest.py to rebuild...')
    print()
    os.system('python ingest.py')

#Reset log command: rebuilds index_log.json by scanning the ChromaDB collection for all indexed sources 
# and their metadata, then tries to match them with files in docs/ to fill in hashes

def cmd_reset_log():
    head('RESET LOG — Rebuild index_log.json from ChromaDB')
    line()

    col   = get_collection()
    count = col.count()

    if count == 0:
        warn('Database is empty. Nothing to rebuild log from.')
        return

    result  = col.get(include=['metadatas'])
    new_log = {}

    for meta in result['metadatas']:
        src = meta.get('source')
        if not src or src in new_log:
            continue
        new_log[src] = {
            'hash':       '',
            'chunks':     meta.get('total_chunks', '?'),
            'indexed_at': meta.get('indexed_at', 'unknown'),
            'file_type':  meta.get('file_type', '?'),
            'doc_title':  meta.get('doc_title', src),
        }

    # Try to fill in hashes from actual files
    for fp in get_docs_in_folder():
        if fp.name in new_log:
            new_log[fp.name]['hash'] = file_hash(str(fp))

    save_index_log(new_log)
    ok(f'Rebuilt index_log.json with {len(new_log)} record(s).')
    for name in sorted(new_log.keys()):
        print(f'      📑  {name}')

# Main entry point: parse command-line arguments and call the appropriate function

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MUM Chatbot — Database Maintenance Tool',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--status',    action='store_true', help='Show full database health report')
    parser.add_argument('--clean',     action='store_true', help='Remove chunks for deleted/missing files')
    parser.add_argument('--remove',    metavar='FILE',      help='Remove one specific document by filename')
    parser.add_argument('--backup',    action='store_true', help='Backup the database to a zip file')
    parser.add_argument('--rebuild',   action='store_true', help='Wipe and rebuild everything from scratch')
    parser.add_argument('--reset-log', action='store_true', help='Rebuild index_log.json from ChromaDB')

    args = parser.parse_args()

    if args.status:
        cmd_status()
    elif args.clean:
        cmd_clean()
    elif args.remove:
        cmd_remove(args.remove)
    elif args.backup:
        cmd_backup()
    elif args.rebuild:
        cmd_rebuild()
    elif args.reset_log:
        cmd_reset_log()
    else:
        print(f'\n{BOLD}MUM Chatbot — Database Maintenance Tool{RESET}')
        print('Usage:')
        print('  python maintain.py --status              Full health report')
        print('  python maintain.py --clean               Remove stale/deleted document chunks')
        print('  python maintain.py --remove filename.pdf Remove one document')
        print('  python maintain.py --backup              Backup database to zip')
        print('  python maintain.py --rebuild             Wipe and rebuild everything')
        print('  python maintain.py --reset-log           Rebuild index_log.json from DB\n')