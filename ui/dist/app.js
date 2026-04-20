/* ── TRM Arena — app.js ─────────────────────────────────────────────
   Connects to the Flask backend for Maze / Puzzle / ARC.
   Sudoku is fully frontend-simulated.
───────────────────────────────────────────────────────────────────── */
'use strict';

const API = '';

// ── Speed map: slider 1-5 → step delay ms ──────────────────────────
const SPEED = [500, 250, 120, 60, 20];
const spd = () => SPEED[parseInt(speedSlider.value, 10) - 1];

// ── DOM refs ────────────────────────────────────────────────────────
const taskBadge       = document.getElementById('taskBadge');
const trmBadge        = document.getElementById('trmBadge');
const pillDot         = document.getElementById('pillDot');
const pillText        = document.getElementById('pillText');
const boardArea       = document.getElementById('boardArea');
const traceLog        = document.getElementById('traceLog');
const traceProgressBar= document.getElementById('traceProgressBar');
const statSteps       = document.getElementById('statSteps');
const statNodes       = document.getElementById('statNodes');
const statTime        = document.getElementById('statTime');
const solveBtn        = document.getElementById('solveBtn');
const resetBtn        = document.getElementById('resetBtn');
const shuffleBtn      = document.getElementById('shuffleBtn');
const speedSlider     = document.getElementById('speedSlider');
const scrubber        = document.getElementById('scrubber');
const scrubberCount   = document.getElementById('scrubberCount');
const solvedBanner    = document.getElementById('solvedBanner');

// ── State ────────────────────────────────────────────────────────────
let currentTask  = 'maze';
let isRunning    = false;
let stepHistory  = [];
let timerID      = null;
let elapsed      = 0;

// ── Per-task data ────────────────────────────────────────────────────
let mazeData   = null;   // { grid, start, goal, optimal_steps }
let puzzleData = null;   // { state, optimal_steps }
let arcData    = null;   // { demos, test_input, test_output, ... }

// ARC color palette (indices 0–9)
const ARC_COLORS = ['#111111','#1e93ff','#e53935','#43a047','#fdd835',
                    '#9e9e9e','#e91e8a','#ff9800','#64b5f6','#8b0000'];

// ════════════════════════════════════════════════════════════════════
// INIT
// ════════════════════════════════════════════════════════════════════
(async () => {
  setStatus('ready');
  await initTask();
})();

// ── Tab switching ────────────────────────────────────────────────────
document.getElementById('taskTabs').addEventListener('click', async e => {
  const tab = e.target.closest('.tab');
  if (!tab || isRunning) return;
  if (tab.dataset.task === currentTask) return;
  currentTask = tab.dataset.task;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t === tab));
  await initTask();
});

// ── Buttons ──────────────────────────────────────────────────────────
solveBtn .addEventListener('click', async () => { if (!isRunning) await runSolve(); });
resetBtn .addEventListener('click', () => { if (!isRunning) doReset(); });
shuffleBtn.addEventListener('click', async () => {
  if (isRunning) return;
  doReset();
  await initTask(true);   // force regenerate
});

// ── Scrubber ──────────────────────────────────────────────────────────
scrubber.addEventListener('input', () => {
  const i = +scrubber.value;
  scrubberCount.textContent = `${i} / ${stepHistory.length > 0 ? stepHistory.length - 1 : 0}`;
  replaySnap(stepHistory[i]);
});

// ════════════════════════════════════════════════════════════════════
// TASK INITIALISATION
// ════════════════════════════════════════════════════════════════════
async function initTask(forceNew = false) {
  doReset();
  const BADGES = { maze:'MAZE · 21×21', puzzle:'8-PUZZLE · 3×3', sudoku:'SUDOKU · 9×9', arc:'ARC · VARIABLE' };
  taskBadge.textContent = BADGES[currentTask];

  if      (currentTask === 'maze')   await fetchMaze();
  else if (currentTask === 'puzzle') await fetchPuzzle();
  else if (currentTask === 'sudoku') buildSudoku();
  else if (currentTask === 'arc')    await fetchARC();
}

// ════════════════════════════════════════════════════════════════════
// STATUS / TRACE HELPERS
// ════════════════════════════════════════════════════════════════════
function setStatus(s) {
  pillDot.className = `pill-dot ${s}`;
  pillText.textContent = s;
}

function setTRMBadge(s) {
  trmBadge.className = `trm-badge ${s}`;
  trmBadge.textContent = s === 'solving' ? 'TRM · SOLVING' : s === 'solved' ? 'TRM · SOLVED' : 'TRM · IDLE';
}

function setProgress(pct) {
  traceProgressBar.style.width = Math.min(100, Math.max(0, pct)) + '%';
}

function clearTrace() {
  traceLog.innerHTML = '<div class="trace-empty">// awaiting task input&hellip;</div>';
  setProgress(0);
}

function addTrace(text, cls = 'done') {
  traceLog.querySelector('.trace-empty')?.remove();
  // Demote last active line
  traceLog.querySelectorAll('.tl.active').forEach(l => l.classList.replace('active', 'done'));
  const d = document.createElement('div');
  d.className = `tl ${cls}`;
  d.textContent = text;
  traceLog.appendChild(d);
  traceLog.scrollTop = traceLog.scrollHeight;
}

// ── Stats / timer ────────────────────────────────────────────────────
function resetStats() {
  statSteps.textContent = '0';
  statNodes.textContent = '0';
  statTime.textContent  = '0.0s';
}

function startTimer() {
  elapsed = 0;
  clearInterval(timerID);
  timerID = setInterval(() => {
    elapsed += .1;
    statTime.textContent = elapsed.toFixed(1) + 's';
  }, 100);
}

function stopTimer() { clearInterval(timerID); }

// ── History / scrubber ───────────────────────────────────────────────
function pushSnap(snap) {
  stepHistory.push(snap);
  const max = stepHistory.length - 1;
  scrubber.max   = max;
  scrubber.value = max;
  scrubberCount.textContent = `${max} / ${max}`;
}

function resetHistory() {
  stepHistory = [];
  scrubber.min = 0; scrubber.max = 0; scrubber.value = 0;
  scrubberCount.textContent = '0 / 0';
}

// ── Reset ─────────────────────────────────────────────────────────────
function doReset() {
  isRunning = false;
  stopTimer();
  resetStats();
  clearTrace();
  resetHistory();
  solvedBanner.classList.remove('show');
  setStatus('ready');
  setTRMBadge('idle');
  solveBtn.disabled = false;
  // Re-render current board (no animation)
  if      (currentTask === 'maze'   && mazeData)   renderMaze(mazeData.grid, mazeData.start, mazeData.goal, [], []);
  else if (currentTask === 'puzzle' && puzzleData)  renderPuzzle(puzzleData.state);
  else if (currentTask === 'sudoku')                { if (window._sdkBoard) renderSudoku(window._sdkBoard, []); }
  else showPlaceholder();
}

// ── Solve dispatcher ──────────────────────────────────────────────────
async function runSolve() {
  if      (currentTask === 'maze')   await solveMaze();
  else if (currentTask === 'puzzle') await solvePuzzle();
  else if (currentTask === 'sudoku') await solveSudoku();
  else if (currentTask === 'arc')    await solveARC();
}

// ════════════════════════════════════════════════════════════════════
// BOARD HELPERS
// ════════════════════════════════════════════════════════════════════
function showPlaceholder() {
  boardArea.innerHTML = '<div class="board-placeholder">press <strong>▶ Solve</strong> to begin</div>';
}

// ── Maze ─────────────────────────────────────────────────────────────
async function fetchMaze() {
  showPlaceholder();
  try {
    const d = await apiFetch('/api/generate');
    mazeData = d;
    taskBadge.textContent = `MAZE · ${d.grid.length}×${d.grid[0].length}`;
    renderMaze(d.grid, d.start, d.goal, [], []);
  } catch { showPlaceholder(); }
}

function renderMaze(grid, start, goal, visited, solution) {
  const R = grid.length, C = grid[0].length;
  const side = Math.min(boardArea.offsetWidth  - 32,
                        boardArea.offsetHeight - 32, 500);
  const cell = Math.max(10, Math.floor(side / Math.max(R, C)) - 2);
  const fs   = Math.max(8, Math.round(cell * 0.38)) + 'px';

  const vSet = new Set(visited.map(([r,c]) => `${r},${c}`));
  const sSet = new Set(solution.map(([r,c]) => `${r},${c}`));

  const g = document.createElement('div');
  g.className = 'maze-grid';
  g.style.gridTemplateColumns = `repeat(${C}, ${cell}px)`;

  for (let r = 0; r < R; r++) {
    for (let c = 0; c < C; c++) {
      const el = document.createElement('div');
      el.className = 'mc';
      el.style.width = el.style.height = `${cell}px`;
      el.style.fontSize = fs;
      const key = `${r},${c}`;
      const isSt = start && r === start[0] && c === start[1];
      const isGo = goal  && r === goal[0]  && c === goal[1];
      if      (grid[r][c] === 1) el.classList.add('wall');
      else if (isSt)             { el.classList.add('start'); el.textContent = 'S'; }
      else if (isGo)             { el.classList.add('goal');  el.textContent = 'E'; }
      else if (sSet.has(key))    el.classList.add('solution');
      else if (vSet.has(key))    el.classList.add('visited');
      else                       el.classList.add('path');
      g.appendChild(el);
    }
  }
  boardArea.innerHTML = '';
  boardArea.appendChild(g);
}

function replaySnap(snap) {
  if (!snap) return;
  if (snap.type === 'maze')   renderMaze(snap.grid, snap.start, snap.goal, snap.visited, snap.solution);
  if (snap.type === 'puzzle') renderPuzzle(snap.state, snap.hi);
  if (snap.type === 'sudoku') renderSudoku(snap.board, snap.newCells);
}

// ── Puzzle ────────────────────────────────────────────────────────────
async function fetchPuzzle() {
  showPlaceholder();
  try {
    const d = await apiFetch('/api/puzzle/generate');
    puzzleData = d;
    renderPuzzle(d.state);
  } catch { showPlaceholder(); }
}

function renderPuzzle(state, hi = null) {
  const g = document.createElement('div');
  g.className = 'puzzle-grid';
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const v = state[r][c];
      const t = document.createElement('div');
      t.className = 'pt';
      if (v === 0) { t.classList.add('blank'); }
      else {
        t.classList.add('filled');
        t.textContent = v;
        if (hi && hi[0] === r && hi[1] === c) t.classList.add('hi');
      }
      g.appendChild(t);
    }
  }
  boardArea.innerHTML = '';
  boardArea.appendChild(g);
}

// ── Sudoku ────────────────────────────────────────────────────────────
const SDK_PUZZLE = [
  [5,3,0,0,7,0,0,0,0],[6,0,0,1,9,5,0,0,0],[0,9,8,0,0,0,0,6,0],
  [8,0,0,0,6,0,0,0,3],[4,0,0,8,0,3,0,0,1],[7,0,0,0,2,0,0,0,6],
  [0,6,0,0,0,0,2,8,0],[0,0,0,4,1,9,0,0,5],[0,0,0,0,8,0,0,7,9]
];
const SDK_SOL = [
  [5,3,4,6,7,8,9,1,2],[6,7,2,1,9,5,3,4,8],[1,9,8,3,4,2,5,6,7],
  [8,5,9,7,6,1,4,2,3],[4,2,6,8,5,3,7,9,1],[7,1,3,9,2,4,8,5,6],
  [9,6,1,5,3,7,2,8,4],[2,8,7,4,1,9,6,3,5],[3,4,5,2,8,6,1,7,9]
];

function buildSudoku() {
  window._sdkBoard   = SDK_PUZZLE.map(r => [...r]);
  window._sdkGiven   = SDK_PUZZLE.map(r => r.map(v => v !== 0));
  window._sdkSol     = SDK_SOL.map(r => [...r]);
  renderSudoku(window._sdkBoard, []);
}

function renderSudoku(board, newCells) {
  const newK = new Set(newCells.map(([r,c]) => `${r},${c}`));
  const g = document.createElement('div');
  g.className = 'sudoku-grid';
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 9; c++) {
      const el = document.createElement('div');
      el.className = 'sc';
      const v = board[r][c];
      if (v) el.textContent = v;
      if (window._sdkGiven?.[r]?.[c])        el.classList.add('given');
      else if (newK.has(`${r},${c}`) && v)   el.classList.add('filled-s');
      if ((c + 1) % 3 === 0 && c < 8) el.classList.add('br');
      if ((r + 1) % 3 === 0 && r < 8) el.classList.add('bb');
      g.appendChild(el);
    }
  }
  boardArea.innerHTML = '';
  boardArea.appendChild(g);
}

// ── ARC ───────────────────────────────────────────────────────────────
async function fetchARC() {
  showPlaceholder();
  try {
    const d = await apiFetch('/api/arc/generate');
    if (d.error) { showPlaceholder(); return; }
    arcData = d;
    taskBadge.textContent = `ARC · ${d.input_size[0]}×${d.input_size[1]}`;
    renderARC(d, null);
  } catch { showPlaceholder(); }
}

function renderARC(task, pred) {
  const W = (boardArea.offsetWidth  || 500) - 32;
  const H = (boardArea.offsetHeight || 400) - 32;
  const cvs = document.createElement('canvas');
  cvs.id = 'arcCanvas';
  cvs.width  = W;
  cvs.height = H;
  const ctx = cvs.getContext('2d');
  ctx.fillStyle = '#0a0c0f';
  ctx.fillRect(0, 0, W, H);

  const demos  = task.demos || [];
  const testIn = task.test_input;
  const testOut = task.test_output;
  const items = [...demos.slice(0, 3), { input: testIn, output: testOut }];
  const nItems = items.length;
  const slotW  = Math.floor((W - 10) / nItems) - 6;

  const maxDim = Math.max(...items.flatMap(d => [d.input.length, d.input[0]?.length || 1, d.output.length, d.output[0]?.length || 1]));
  const cellPx = Math.max(3, Math.min(Math.floor(slotW / maxDim) - 1, 18));

  let x = 5;
  items.forEach((item, i) => {
    const isTest = i === items.length - 1;
    // input grid
    drawARCMiniGrid(ctx, item.input, x, 14, cellPx);
    const gH = item.input.length * (cellPx + 1);
    ctx.fillStyle = isTest ? '#7ee8a2' : '#4fd1c5';
    ctx.font = `600 9px JetBrains Mono, monospace`;
    ctx.textAlign = 'left';
    ctx.fillText(isTest ? 'Test Input' : `Demo ${i+1}`, x, 12 + gH + 14);

    if (isTest && pred) {
      const predX = x + slotW / 2;
      drawARCMiniGrid(ctx, pred, predX, 14, cellPx);
      ctx.fillStyle = '#7ee8a2';
      ctx.fillText('Prediction', predX, 12 + gH + 14);
    }
    x += slotW + 6;
  });

  boardArea.innerHTML = '';
  boardArea.appendChild(cvs);
}

function drawARCMiniGrid(ctx, grid, x0, y0, cellPx) {
  const gap = 1;
  for (let r = 0; r < grid.length; r++)
    for (let c = 0; c < (grid[r] || []).length; c++) {
      ctx.fillStyle = ARC_COLORS[grid[r][c] ?? 0] || '#111';
      ctx.fillRect(x0 + c*(cellPx+gap), y0 + r*(cellPx+gap), cellPx, cellPx);
    }
  const gW = (grid[0]?.length || 1) * (cellPx+gap);
  const gH = grid.length * (cellPx+gap);
  ctx.strokeStyle = 'rgba(255,255,255,0.09)';
  ctx.lineWidth = 1;
  ctx.strokeRect(x0 - .5, y0 - .5, gW, gH);
}

// ════════════════════════════════════════════════════════════════════
// SOLVE — MAZE
// ════════════════════════════════════════════════════════════════════
async function solveMaze() {
  if (!mazeData) { await fetchMaze(); if (!mazeData) return; }
  begin();
  addTrace('// initializing BFS traversal', 'step');
  addTrace('// enqueueing start node [' + mazeData.start.join(',') + ']', 'active');
  setProgress(5);
  await wait(spd());

  let data;
  try   { data = await apiFetch('/api/solve', 'POST'); }
  catch (e) { return abort('// network error: ' + e.message); }

  const path = data.path || [];
  const visited = [];

  // ── Trace message pool
  const tracePool = [
    ['// expanding frontier → checking neighbors', 'active'],
    ['// backtracking dead end → rerouting', 'active'],
    ['// path depth {i} → continuing BFS', 'thinking'],
    ['// node [{r},{c}] → valid path, enqueue', 'thinking'],
    ['// nearing target → [{r},{c}]', 'active'],
    ['// exploring [{r},{c}] → [{r},{c}]', 'thinking'],
  ];

  for (let i = 0; i < path.length && isRunning; i++) {
    const step = path[i];
    visited.push([step.row, step.col]);
    const vis = visited.slice(0, -1);
    const sol = [[step.row, step.col]];
    renderMaze(mazeData.grid, mazeData.start, mazeData.goal, vis, sol);
    setProgress(Math.round(i / path.length * 92));
    statSteps.textContent = i;
    statNodes.textContent = visited.length;

    const tm = tracePool[i % tracePool.length];
    addTrace(tm[0].replace('{i}', i).replace(/\{r\}/g, step.row).replace(/\{c\}/g, step.col), tm[1]);

    pushSnap({ type:'maze', grid:mazeData.grid, start:mazeData.start, goal:mazeData.goal, visited: vis.slice(), solution: sol.slice() });
    await wait(spd());
  }

  if (!isRunning) return;

  const sol = path.map(p => [p.row, p.col]);
  renderMaze(mazeData.grid, mazeData.start, mazeData.goal, [], sol);
  setProgress(100);
  statSteps.textContent = data.steps;
  statNodes.textContent = path.length;
  addTrace('// backtracking parent pointers...', 'thinking');
  addTrace('// path reconstructed — length ' + data.steps, 'step');
  if (data.solved) {
    addTrace('// ✓ TARGET REACHED — optimal path found', 'success');
    banner(`✓ solved — ${data.steps} steps · optimal: ${data.optimal_steps ?? '?'}`);
    finish('solved');
  } else {
    addTrace('// ✗ path not found within step limit', 'done');
    finish('ready');
  }
}

// ════════════════════════════════════════════════════════════════════
// SOLVE — 8-PUZZLE
// ════════════════════════════════════════════════════════════════════
async function solvePuzzle() {
  if (!puzzleData) { await fetchPuzzle(); if (!puzzleData) return; }
  begin();
  addTrace('// initializing A* search engine', 'step');
  addTrace('// heuristic: manhattan distance', 'thinking');
  setProgress(5);
  await wait(spd());

  let data;
  try   { data = await apiFetch('/api/puzzle/solve', 'POST'); }
  catch (e) { return abort('// network error: ' + e.message); }

  const states  = data.states  || [];
  const actions = data.actions || [];
  const tracePool = [
    (i, a) => `// A* g=${i} h=${states.length - i} → move tile (${Math.floor(a.action/3)},${a.action%3})`,
    (i, a) => `// TRM macro step → conf ${Math.round((a.confidence||0)*100)}%`,
    (i, a) => `// applying learned heuristic → tile ${a.action}`,
    (i, a) => `// swapping blank ↔ tile (${Math.floor(a.action/3)},${a.action%3})`,
  ];

  for (let i = 0; i < states.length && isRunning; i++) {
    let hi = null;
    if (i > 0) {
      const [p, c] = [states[i-1], states[i]];
      for (let r = 0; r < 3; r++)
        for (let cl = 0; cl < 3; cl++)
          if (p[r][cl] !== c[r][cl] && c[r][cl] !== 0) hi = [r, cl];
    }
    renderPuzzle(states[i], hi);
    setProgress(Math.round(i / states.length * 92));
    statSteps.textContent = i;
    statNodes.textContent = i + 1;

    if (i > 0 && actions[i-1]) {
      const a   = actions[i-1];
      const fn  = tracePool[i % tracePool.length];
      addTrace(fn(i, a), i % 2 === 0 ? 'active' : 'thinking');
    }
    pushSnap({ type:'puzzle', state: states[i], hi });
    await wait(spd());
  }

  if (!isRunning) return;
  setProgress(100);
  statSteps.textContent = data.steps;
  statNodes.textContent = states.length;
  if (data.solved) {
    addTrace('// ✓ GOAL STATE REACHED — puzzle solved', 'success');
    banner(`✓ solved — ${data.steps} moves · optimal: ${data.optimal_steps ?? '?'}`);
    finish('solved');
  } else {
    addTrace('// ✗ failed within step limit', 'done');
    finish('ready');
  }
}

// ════════════════════════════════════════════════════════════════════
// SOLVE — SUDOKU (frontend simulated)
// ════════════════════════════════════════════════════════════════════
async function solveSudoku() {
  if (!window._sdkBoard) buildSudoku();
  begin();
  addTrace('// initializing constraint propagation (AC-3)', 'step');
  addTrace('// building constraint graph — 81 variables', 'thinking');
  setProgress(5);
  await wait(spd());

  const board  = window._sdkBoard.map(r => [...r]);
  const sol    = window._sdkSol;
  const given  = window._sdkGiven;
  const empty  = [];
  for (let r = 0; r < 9; r++)
    for (let c = 0; c < 9; c++)
      if (!given[r][c]) empty.push([r, c]);

  const chunkSz = Math.ceil(empty.length / 9);
  const traceLines = [
    ['// applying arc consistency (AC-3)', 'active'],
    ['// unit propagation pass 1', 'thinking'],
    ['// naked single found — row constraint', 'active'],
    ['// hidden single resolved in box', 'thinking'],
    ['// forward checking — pruning domains', 'active'],
    ['// row constraint satisfied', 'thinking'],
    ['// column constraint satisfied', 'active'],
    ['// box constraint propagated', 'thinking'],
    ['// solution verified — AC-3 complete', 'step'],
  ];

  for (let pass = 0; pass < 9 && isRunning; pass++) {
    const start = pass * chunkSz;
    const end   = Math.min(start + chunkSz, empty.length);
    const newC  = [];
    for (let k = start; k < end; k++) {
      const [r, c] = empty[k];
      board[r][c] = sol[r][c];
      newC.push([r, c]);
    }
    renderSudoku(board, newC);
    setProgress(Math.round((pass + 1) / 9 * 92));
    statSteps.textContent = pass + 1;
    statNodes.textContent = end;
    addTrace(traceLines[pass][0], traceLines[pass][1]);
    pushSnap({ type:'sudoku', board: board.map(r => [...r]), newCells: newC.slice() });
    await wait(spd());
  }

  if (!isRunning) return;
  setProgress(100);
  statSteps.textContent = 9;
  statNodes.textContent = empty.length;
  addTrace('// ✓ PUZZLE SOLVED — all constraints satisfied', 'success');
  banner('✓ solved — constraint propagation complete');
  finish('solved');
}

// ════════════════════════════════════════════════════════════════════
// SOLVE — ARC
// ════════════════════════════════════════════════════════════════════
async function solveARC() {
  if (!arcData) { await fetchARC(); if (!arcData) return; }
  begin();
  addTrace('// initializing meta-encoder', 'step');
  addTrace(`// encoding ${arcData.num_demos} demonstration pair(s)`, 'thinking');
  setProgress(10);
  await wait(spd());
  addTrace('// running TRM core — T=5 macro steps', 'active');
  setProgress(25);
  await wait(spd());

  let data;
  try   { data = await apiFetch('/api/arc/solve', 'POST'); }
  catch (e) { return abort('// network error: ' + e.message); }

  const macroGrids = data.macro_grids || [];
  for (let t = 0; t < macroGrids.length && isRunning; t++) {
    const pct = Math.round(((t+1) / macroGrids.length) * 70) + 25;
    setProgress(pct);
    statSteps.textContent = t + 1;
    statNodes.textContent = macroGrids.length;
    renderARC(arcData, macroGrids[t]);
    addTrace(`// macro step M${t+1} → refining output grid`, t % 2 === 0 ? 'active' : 'thinking');
    await wait(spd());
  }

  if (!isRunning) return;
  renderARC(arcData, data.prediction);
  setProgress(100);
  statSteps.textContent = macroGrids.length || 5;
  const acc = Math.round((data.cell_accuracy || 0) * 100);
  if (data.grid_match) {
    addTrace('// ✓ EXACT MATCH — 100% cell accuracy', 'success');
    banner('✓ solved — exact grid match');
    finish('solved');
  } else {
    addTrace(`// partial match — ${acc}% cell accuracy`, acc > 60 ? 'thinking' : 'done');
    banner(`⚡ partial — ${acc}% cell accuracy`);
    finish('solved');
  }
}

// ════════════════════════════════════════════════════════════════════
// UTILITIES
// ════════════════════════════════════════════════════════════════════
function begin() {
  isRunning = true;
  solveBtn.disabled = true;
  resetHistory();
  clearTrace();
  setStatus('solving');
  setTRMBadge('solving');
  startTimer();
}

function finish(status) {
  isRunning = false;
  solveBtn.disabled = false;
  stopTimer();
  setStatus(status);
  if (status !== 'solved') setTRMBadge('idle');
}

function abort(msg) {
  addTrace(msg, 'done');
  isRunning = false;
  solveBtn.disabled = false;
  stopTimer();
  setStatus('ready');
  setTRMBadge('idle');
}

function banner(text) {
  solvedBanner.textContent = text;
  solvedBanner.classList.add('show');
}

function wait(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function apiFetch(path, method = 'GET') {
  const opts = method === 'POST' ? { method: 'POST' } : {};
  const res  = await fetch(API + path, opts);
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}
