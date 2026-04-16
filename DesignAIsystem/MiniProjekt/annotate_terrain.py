"""
Terrain annotation web app for Kingdomino.

Run:  python annotate_terrain.py
Open: http://localhost:5000
"""

import os
import json
import cv2 as cv
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

# ── config ─────────────────────────────────────────────────────────────────────
image_folder  = r"C:\Users\110492\Github\MinRepo\DesignAIsystem\MiniProjekt\dataset"
tiles_folder  = r"C:\Users\110492\Github\MinRepo\DesignAIsystem\MiniProjekt\tiles"
output_file   = r"C:\Users\110492\Github\MinRepo\DesignAIsystem\MiniProjekt\ground_truth_terrain.json"

tile_size  = 100
grid_size  = 5

terrain_types = ['wheat', 'forest', 'water', 'grass', 'swamp', 'mine', 'castle', 'unknown']

terrain_colors = {
    'wheat':   '#F5C842',
    'forest':  '#2D6A2D',
    'water':   '#1565C0',
    'grass':   '#66BB6A',
    'swamp':   '#827717',
    'mine':    '#757575',
    'castle':  '#BDBDBD',
    'unknown': '#424242',
}

# HSV ranges per terrain type — list of (lower, upper) pairs
hsv_ranges = {
    'wheat':  [(np.array([15,  60, 100]), np.array([35, 255, 255]))],
    'forest': [(np.array([35,  50,  20]), np.array([85, 255, 150]))],
    'water':  [(np.array([90,  60,  60]), np.array([140, 255, 255]))],
    'grass':  [(np.array([35,  40, 100]), np.array([85,  255, 255]))],
    'swamp':  [(np.array([10,  30,  30]), np.array([40,  150, 130]))],
    'mine':   [(np.array([0,    0,  30]), np.array([180,  40, 130]))],
}
# ───────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)


def classify_tile(tile):
    """Return the best-guess terrain label and pixel-match score for a tile."""
    hsv = cv.cvtColor(tile, cv.COLOR_BGR2HSV)
    best_label = 'unknown'
    best_score = 0

    for label, ranges in hsv_ranges.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv.inRange(hsv, lower, upper)
        score = int(np.sum(mask > 0))
        if score > best_score:
            best_score = score
            best_label = label

    # Castle / home tile: low saturation + high brightness
    sat = float(hsv[:, :, 1].mean())
    val = float(hsv[:, :, 2].mean())
    if sat < 30 and val > 150:
        best_label = 'castle'

    return best_label, best_score


def load_labels():
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_labels(labels):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)


def build_board_data(image_id, img, labels):
    """Build a list of cell dicts for one board, pre-filling with HSV guesses."""
    cells = []
    for y in range(grid_size):
        for x in range(grid_size):
            tile_id = f"{image_id}_x{x}_y{y}"
            tile = img[y * tile_size:(y + 1) * tile_size,
                       x * tile_size:(x + 1) * tile_size]
            guess, score = classify_tile(tile)
            confirmed = tile_id in labels
            label = labels.get(tile_id, guess)
            cells.append({
                'tile_id':   tile_id,
                'x':         x,
                'y':         y,
                'label':     label,
                'guess':     guess,
                'score':     score,
                'confirmed': confirmed,
            })
    return cells


def load_all_boards():
    labels = load_labels()
    boards = []
    for filename in sorted(
        os.listdir(image_folder),
        key=lambda f: int(os.path.splitext(f)[0])
              if os.path.splitext(f)[0].isdigit() else 0
    ):
        if not filename.lower().endswith(('.jpg', '.png')):
            continue
        image_id = os.path.splitext(filename)[0]
        img = cv.imread(os.path.join(image_folder, filename))
        if img is None:
            continue
        cells = build_board_data(image_id, img, labels)
        confirmed_count = sum(1 for c in cells if c['confirmed'])
        boards.append({
            'image_id':        image_id,
            'cells':           cells,
            'confirmed_count': confirmed_count,
        })
    return boards, labels


# ── routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    boards, _ = load_all_boards()
    total_tiles     = sum(len(b['cells']) for b in boards)
    confirmed_total = sum(b['confirmed_count'] for b in boards)

    # Serialize boards to JSON for the frontend
    boards_json = json.dumps(boards)
    colors_json = json.dumps(terrain_colors)
    types_json  = json.dumps(terrain_types)

    return render_page(boards_json, colors_json, types_json,
                       total_tiles, confirmed_total)


@app.route('/tiles/<path:filename>')
def serve_tile(filename):
    return send_from_directory(tiles_folder, filename)


@app.route('/api/label', methods=['POST'])
def update_label():
    data    = request.get_json()
    tile_id = data.get('tile_id')
    label   = data.get('label')
    if not tile_id or label not in terrain_types:
        return jsonify({'ok': False, 'error': 'invalid input'}), 400
    labels = load_labels()
    labels[tile_id] = label
    save_labels(labels)
    confirmed = sum(1 for v in labels.values() if v != 'unknown')
    return jsonify({'ok': True, 'confirmed': confirmed})


@app.route('/api/approve_board', methods=['POST'])
def approve_board():
    """Accept all HSV guesses for one board."""
    data     = request.get_json()
    image_id = data.get('image_id')
    labels   = load_labels()
    img      = cv.imread(os.path.join(image_folder, f"{image_id}.jpg"))
    if img is None:
        return jsonify({'ok': False, 'error': 'image not found'}), 404
    for y in range(grid_size):
        for x in range(grid_size):
            tile_id = f"{image_id}_x{x}_y{y}"
            if tile_id not in labels:
                tile  = img[y * tile_size:(y + 1) * tile_size,
                            x * tile_size:(x + 1) * tile_size]
                guess, _ = classify_tile(tile)
                labels[tile_id] = guess
    save_labels(labels)
    return jsonify({'ok': True})


@app.route('/api/approve_all', methods=['POST'])
def approve_all():
    """Accept all HSV guesses for every board."""
    labels = load_labels()
    for filename in os.listdir(image_folder):
        if not filename.lower().endswith(('.jpg', '.png')):
            continue
        image_id = os.path.splitext(filename)[0]
        img = cv.imread(os.path.join(image_folder, filename))
        if img is None:
            continue
        for y in range(grid_size):
            for x in range(grid_size):
                tile_id = f"{image_id}_x{x}_y{y}"
                if tile_id not in labels:
                    tile  = img[y * tile_size:(y + 1) * tile_size,
                                x * tile_size:(x + 1) * tile_size]
                    guess, _ = classify_tile(tile)
                    labels[tile_id] = guess
    save_labels(labels)
    return jsonify({'ok': True, 'total': len(labels)})


# ── HTML template ──────────────────────────────────────────────────────────────

def render_page(boards_json, colors_json, types_json, total_tiles, confirmed_total):
    pct = int(confirmed_total / total_tiles * 100) if total_tiles else 0
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Kingdomino Terrain Annotator</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee; }}

  header {{
    position: sticky; top: 0; z-index: 100;
    background: #16213e; padding: 12px 20px;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    border-bottom: 1px solid #0f3460;
  }}
  header h1 {{ font-size: 1.1rem; white-space: nowrap; }}

  .progress-wrap {{ flex: 1; min-width: 160px; }}
  .progress-bar {{
    height: 10px; border-radius: 5px; background: #0f3460;
    overflow: hidden; margin-top: 4px;
  }}
  .progress-fill {{
    height: 100%; background: #e94560;
    transition: width 0.3s;
    width: {pct}%;
  }}
  .progress-label {{ font-size: 0.75rem; color: #aaa; }}

  .btn {{
    padding: 6px 14px; border: none; border-radius: 6px;
    cursor: pointer; font-size: 0.8rem; font-weight: 600;
    transition: opacity 0.15s;
  }}
  .btn:hover {{ opacity: 0.85; }}
  .btn-approve-all {{ background: #e94560; color: #fff; }}
  .btn-filter {{ background: #0f3460; color: #eee; }}
  .btn-filter.active {{ background: #e94560; }}

  .legend {{
    display: flex; gap: 8px; flex-wrap: wrap; font-size: 0.72rem;
  }}
  .legend-item {{
    display: flex; align-items: center; gap: 4px;
  }}
  .legend-dot {{
    width: 10px; height: 10px; border-radius: 3px; flex-shrink: 0;
  }}

  main {{ padding: 16px 20px; }}

  .boards-grid {{
    display: flex; flex-wrap: wrap; gap: 20px;
  }}

  .board-card {{
    background: #16213e;
    border: 1px solid #0f3460;
    border-radius: 10px;
    padding: 10px;
    width: fit-content;
  }}
  .board-card.done {{ border-color: #2d6a2d; }}

  .board-header {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 8px; gap: 8px;
  }}
  .board-title {{ font-size: 0.8rem; font-weight: 600; color: #aaa; }}
  .btn-approve {{
    background: #2d6a2d; color: #fff;
    font-size: 0.7rem; padding: 3px 8px;
  }}

  .grid {{
    display: grid;
    grid-template-columns: repeat(5, 60px);
    grid-template-rows:    repeat(5, 60px);
    gap: 2px;
  }}

  .cell {{
    position: relative;
    width: 60px; height: 60px;
    cursor: pointer;
    border-radius: 4px;
    overflow: hidden;
    border: 2px solid transparent;
    transition: border-color 0.15s;
  }}
  .cell:hover {{ border-color: #fff; }}
  .cell img {{
    width: 100%; height: 100%;
    display: block; object-fit: cover;
  }}
  .cell-label {{
    position: absolute; bottom: 0; left: 0; right: 0;
    font-size: 0.55rem; font-weight: 700;
    text-align: center; padding: 1px 0;
    color: #fff; text-shadow: 0 0 3px #000;
    background: rgba(0,0,0,0.45);
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }}
  .cell.confirmed .cell-label {{ background: rgba(0,0,0,0.7); }}
  .cell-dot {{
    position: absolute; top: 3px; right: 3px;
    width: 8px; height: 8px; border-radius: 50%;
  }}
  .cell.confirmed .cell-dot {{ background: #4caf50; }}
  .cell:not(.confirmed) .cell-dot {{ background: #ff9800; }}

  /* popup selector */
  .popup-overlay {{
    display: none;
    position: fixed; inset: 0; z-index: 200;
    background: rgba(0,0,0,0.6);
    align-items: center; justify-content: center;
  }}
  .popup-overlay.open {{ display: flex; }}
  .popup {{
    background: #16213e;
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 20px;
    min-width: 280px;
  }}
  .popup h3 {{ font-size: 0.9rem; margin-bottom: 14px; color: #ccc; }}
  .terrain-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }}
  .terrain-btn {{
    padding: 10px 8px;
    border: 2px solid transparent;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.8rem;
    font-weight: 600;
    color: #fff;
    text-transform: capitalize;
    transition: transform 0.1s;
  }}
  .terrain-btn:hover {{ transform: scale(1.04); }}
  .terrain-btn.selected {{ border-color: #fff; }}
  .popup-cancel {{
    margin-top: 12px; width: 100%;
    background: #0f3460; color: #eee;
  }}
</style>
</head>
<body>

<header>
  <h1>Kingdomino Terrain Annotator</h1>
  <div class="progress-wrap">
    <div class="progress-label" id="prog-label">{confirmed_total} / {total_tiles} confirmed</div>
    <div class="progress-bar"><div class="progress-fill" id="prog-fill"></div></div>
  </div>

  <div class="legend" id="legend"></div>

  <div style="display:flex;gap:6px;flex-wrap:wrap;">
    <button class="btn btn-filter active" onclick="setFilter('all')">All</button>
    <button class="btn btn-filter" onclick="setFilter('incomplete')">Incomplete</button>
    <button class="btn btn-filter" onclick="setFilter('done')">Done</button>
  </div>

  <button class="btn btn-approve-all" onclick="approveAll()">Accept all HSV guesses</button>
</header>

<main>
  <div class="boards-grid" id="boards-grid"></div>
</main>

<!-- Terrain selector popup -->
<div class="popup-overlay" id="popup" onclick="closePopup(event)">
  <div class="popup">
    <h3 id="popup-title">Select terrain</h3>
    <div class="terrain-grid" id="terrain-grid"></div>
    <button class="btn popup-cancel" onclick="closePopup()">Cancel</button>
  </div>
</div>

<script>
const boards      = {boards_json};
const colors      = {colors_json};
const types       = {types_json};
const totalTiles  = {total_tiles};
let confirmed     = {confirmed_total};

let activeTileId  = null;
let activeEl      = null;
let currentFilter = 'all';

// ── Build legend ──────────────────────────────────────────────────────────────
function buildLegend() {{
  const el = document.getElementById('legend');
  types.forEach(t => {{
    el.innerHTML += `<div class="legend-item">
      <div class="legend-dot" style="background:${{colors[t]}}"></div>
      <span>${{t}}</span>
    </div>`;
  }});
}}

// ── Render boards ─────────────────────────────────────────────────────────────
function renderBoards() {{
  const container = document.getElementById('boards-grid');
  container.innerHTML = '';
  boards.forEach(board => {{
    if (currentFilter === 'done'       && board.confirmed_count < 25) return;
    if (currentFilter === 'incomplete' && board.confirmed_count >= 25) return;

    const isDone = board.confirmed_count >= 25;
    const card = document.createElement('div');
    card.className = 'board-card' + (isDone ? ' done' : '');
    card.id = 'board-' + board.image_id;

    card.innerHTML = `
      <div class="board-header">
        <span class="board-title">Board ${{board.image_id}}
          <span id="cnt-${{board.image_id}}" style="color:#888">(${{board.confirmed_count}}/25)</span>
        </span>
        <button class="btn btn-approve" onclick="approveBoard('${{board.image_id}}')">
          Accept guesses
        </button>
      </div>
      <div class="grid" id="grid-${{board.image_id}}"></div>
    `;
    container.appendChild(card);

    const grid = document.getElementById('grid-' + board.image_id);
    board.cells.forEach(cell => {{
      grid.appendChild(buildCell(cell));
    }});
  }});
}}

function buildCell(cell) {{
  const el = document.createElement('div');
  el.className = 'cell' + (cell.confirmed ? ' confirmed' : '');
  el.id = 'cell-' + cell.tile_id;
  el.style.borderColor = colors[cell.label] || '#555';
  el.onclick = () => openPopup(cell.tile_id, cell.label, el);

  el.innerHTML = `
    <img src="/tiles/${{cell.tile_id}}.png" alt="${{cell.tile_id}}" loading="lazy">
    <div class="cell-label" style="background:${{colors[cell.label]}}55">${{cell.label}}</div>
    <div class="cell-dot"></div>
  `;
  return el;
}}

// ── Popup ─────────────────────────────────────────────────────────────────────
function openPopup(tile_id, current_label, cellEl) {{
  activeTileId = tile_id;
  activeEl     = cellEl;
  document.getElementById('popup-title').textContent = tile_id;

  const grid = document.getElementById('terrain-grid');
  grid.innerHTML = '';
  types.forEach(t => {{
    const btn = document.createElement('button');
    btn.className = 'terrain-btn' + (t === current_label ? ' selected' : '');
    btn.style.background = colors[t];
    btn.textContent = t;
    btn.onclick = () => selectTerrain(t);
    grid.appendChild(btn);
  }});

  document.getElementById('popup').classList.add('open');
}}

function closePopup(e) {{
  if (!e || e.target === document.getElementById('popup') || e.target.classList.contains('popup-cancel')) {{
    document.getElementById('popup').classList.remove('open');
    activeTileId = null;
    activeEl     = null;
  }}
}}

function selectTerrain(label) {{
  if (!activeTileId) return;
  fetch('/api/label', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ tile_id: activeTileId, label }})
  }})
  .then(r => r.json())
  .then(data => {{
    if (!data.ok) return;
    // Update cell
    const el = document.getElementById('cell-' + activeTileId);
    if (el) {{
      el.style.borderColor = colors[label];
      el.querySelector('.cell-label').textContent = label;
      el.querySelector('.cell-label').style.background = colors[label] + '55';
      el.classList.add('confirmed');
    }}
    updateProgress(data.confirmed);
    updateBoardCount(activeTileId);
    document.getElementById('popup').classList.remove('open');
  }});
}}

// ── Approve ───────────────────────────────────────────────────────────────────
function approveBoard(image_id) {{
  fetch('/api/approve_board', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ image_id }})
  }})
  .then(r => r.json())
  .then(() => location.reload());
}}

function approveAll() {{
  if (!confirm('Accept HSV guesses for all unannotated tiles?')) return;
  fetch('/api/approve_all', {{ method: 'POST' }})
    .then(r => r.json())
    .then(() => location.reload());
}}

// ── Helpers ───────────────────────────────────────────────────────────────────
function updateProgress(new_confirmed) {{
  confirmed = new_confirmed;
  const pct = Math.round(confirmed / totalTiles * 100);
  document.getElementById('prog-label').textContent =
    confirmed + ' / ' + totalTiles + ' confirmed';
  document.getElementById('prog-fill').style.width = pct + '%';
}}

function updateBoardCount(tile_id) {{
  // tile_id format: imageId_xX_yY
  const image_id = tile_id.split('_x')[0];
  const cnt_el = document.getElementById('cnt-' + image_id);
  if (!cnt_el) return;
  const grid = document.getElementById('grid-' + image_id);
  if (!grid) return;
  const done = grid.querySelectorAll('.cell.confirmed').length;
  cnt_el.textContent = '(' + done + '/25)';
  if (done >= 25) {{
    document.getElementById('board-' + image_id)?.classList.add('done');
  }}
}}

function setFilter(f) {{
  currentFilter = f;
  document.querySelectorAll('.btn-filter').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  renderBoards();
}}

// ── Init ──────────────────────────────────────────────────────────────────────
buildLegend();
renderBoards();
</script>
</body>
</html>"""


if __name__ == '__main__':
    print("Starting annotation server...")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=False, port=5000)
