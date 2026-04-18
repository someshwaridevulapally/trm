// RecursiveNet Task Solver UI

const API_BASE = '';

// Increase canvas resolution for readability
const CANVAS_SIZE = 560;

// DOM Elements
const canvas        = document.getElementById('taskCanvas');
const ctx           = canvas.getContext('2d');
const generateBtn   = document.getElementById('generateBtn');
const solveBtn      = document.getElementById('solveBtn');
const resetBtn      = document.getElementById('resetBtn');
const canvasOverlay = document.getElementById('canvasOverlay');
const statusText    = document.getElementById('statusText');
const optimalStepsEl = document.getElementById('optimalSteps');
const agentStepsEl  = document.getElementById('agentSteps');
const resultStatusEl = document.getElementById('resultStatus');
const gridSizeEl    = document.getElementById('gridSize');
const speedSlider   = document.getElementById('speedSlider');
const speedValue    = document.getElementById('speedValue');
const taskTitle     = document.getElementById('taskTitle');
const taskDescription = document.getElementById('taskDescription');
const navTabs       = document.querySelectorAll('.nav-tab');
const mazeLegend    = document.getElementById('mazeLegend');
const puzzleLegend  = document.getElementById('puzzleLegend');
const arcLegend     = document.getElementById('arcLegend');

// State
let currentTask   = 'maze';
let currentMaze   = null;
let currentStart  = null;
let currentGoal   = null;
let currentPuzzle = null;
let currentARC    = null;   // { demos, test_input, test_output, task_id }
let isAnimating   = false;
let animationSpeed = 80;

// Task configurations
const TASKS = {
    maze: {
        title:       'Maze Solver',
        description: 'Navigate from start to goal using the trained TRM agent',
        gridSize:    21,
        available:   true
    },
    puzzle: {
        title:       '8-Puzzle Solver',
        description: 'Slide tiles to reach goal configuration [1-8, blank] using trained TRM',
        gridSize:    3,
        available:   true
    },
    arc: {
        title:       'ARC-AGI Solver',
        description: 'Learn visual patterns from demos, predict the test output',
        gridSize:    30,
        available:   true
    }
};

// Maze colors
const MAZE_COLORS = {
    wall:       '#0d0d12',
    path:       '#3d3d5c',
    start:      '#10b981',
    goal:       '#f97316',
    agent:      '#8b5cf6',
    agentTrail: 'rgba(139, 92, 246, 0.35)'
};

// Puzzle tile palette
const PUZZLE_TILE_COLORS = [
    'transparent', '#6366f1', '#8b5cf6', '#a855f7', '#ec4899',
    '#f97316', '#eab308', '#22c55e', '#10b981',
];

// ARC-AGI official color palette (indices 0–9)
const ARC_COLORS = [
    '#111111', // 0  black
    '#1e93ff', // 1  blue
    '#e53935', // 2  red
    '#43a047', // 3  green
    '#fdd835', // 4  yellow
    '#9e9e9e', // 5  grey
    '#e91e8a', // 6  pink
    '#ff9800', // 7  orange
    '#64b5f6', // 8  azure
    '#8b0000', // 9  maroon
];

const GOAL_STATE = [[1,2,3],[4,5,6],[7,8,0]];

// Set canvas size
canvas.width  = CANVAS_SIZE;
canvas.height = CANVAS_SIZE;

let GRID_SIZE  = 21;
let CELL_SIZE  = Math.floor(canvas.width / GRID_SIZE);

// -- Event listeners ----------------------------------------------------------

generateBtn.addEventListener('click', generate);
solveBtn.addEventListener('click',    solve);
resetBtn.addEventListener('click',    reset);

speedSlider.addEventListener('input', (e) => {
    animationSpeed = parseInt(e.target.value);
    speedValue.textContent = `${animationSpeed}ms`;
});

navTabs.forEach(tab => {
    tab.addEventListener('click', () => switchTask(tab.dataset.task));
});

// -- Init ---------------------------------------------------------------------

init();

function init() {
    updateTaskUI();
    drawBlankGrid();
}

function switchTask(task) {
    if (task === currentTask || isAnimating) return;
    const cfg = TASKS[task];
    if (!cfg.available) {
        alert(`${cfg.title} is not yet implemented. Coming soon!`);
        return;
    }
    currentTask = task;
    navTabs.forEach(t => t.classList.remove('active'));
    document.querySelector(`[data-task="${task}"]`).classList.add('active');
    reset();
    updateTaskUI();
}

function updateTaskUI() {
    const cfg = TASKS[currentTask];
    taskTitle.textContent       = cfg.title;
    taskDescription.textContent = cfg.description;
    GRID_SIZE = cfg.gridSize;
    CELL_SIZE = Math.floor(canvas.width / GRID_SIZE);

    // Legend visibility
    mazeLegend.style.display    = currentTask === 'maze'   ? '' : 'none';
    puzzleLegend.style.display  = currentTask === 'puzzle' ? '' : 'none';
    arcLegend.style.display     = currentTask === 'arc'    ? '' : 'none';

    // Stats labels
    if (currentTask === 'maze') {
        gridSizeEl.textContent = '10 x 10';
    } else if (currentTask === 'puzzle') {
        gridSizeEl.textContent = '3 x 3';
    } else if (currentTask === 'arc') {
        gridSizeEl.textContent = 'Variable';
    }
}

// -- Blank grid ---------------------------------------------------------------

function drawBlankGrid() {
    if (currentTask === 'puzzle') {
        drawBlankPuzzle();
    } else if (currentTask === 'arc') {
        drawBlankARC();
    } else {
        ctx.fillStyle = MAZE_COLORS.path;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'rgba(255,255,255,0.05)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= GRID_SIZE; i++) {
            ctx.beginPath(); ctx.moveTo(i * CELL_SIZE, 0); ctx.lineTo(i * CELL_SIZE, canvas.height); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, i * CELL_SIZE); ctx.lineTo(canvas.width, i * CELL_SIZE); ctx.stroke();
        }
    }
}

function drawBlankPuzzle() {
    const size = canvas.width, tileW = size / 3;
    ctx.fillStyle = '#1c1c28';
    ctx.fillRect(0, 0, size, size);
    for (let r = 0; r < 3; r++)
        for (let c = 0; c < 3; c++) {
            ctx.fillStyle = '#2a2a3c';
            roundRect(ctx, c * tileW + 4, r * tileW + 4, tileW - 8, tileW - 8, 12);
        }
}

function drawBlankARC() {
    ctx.fillStyle = '#0f0f14';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#1a1a26';
    ctx.font = '14px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('ARC-AGI Task', canvas.width / 2, canvas.height / 2);
}

// -- Maze drawing -------------------------------------------------------------

function drawMaze(maze, start, goal, agentPath = []) {
    ctx.fillStyle = '#0f0f14';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let r = 0; r < maze.length; r++)
        for (let c = 0; c < maze[r].length; c++) {
            ctx.fillStyle = maze[r][c] === 1 ? MAZE_COLORS.wall : MAZE_COLORS.path;
            ctx.fillRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        }

    if (agentPath.length > 1) {
        ctx.fillStyle = MAZE_COLORS.agentTrail;
        for (let i = 0; i < agentPath.length - 1; i++) {
            const s = agentPath[i];
            ctx.fillRect(s.col * CELL_SIZE, s.row * CELL_SIZE, CELL_SIZE, CELL_SIZE);
        }
    }

    if (agentPath.length > 0) {
        const cur = agentPath[agentPath.length - 1];
        const x = cur.col * CELL_SIZE, y = cur.row * CELL_SIZE;
        ctx.fillStyle = MAZE_COLORS.agent;
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
        ctx.beginPath();
        ctx.arc(x + CELL_SIZE / 2, y + CELL_SIZE / 2, CELL_SIZE / 3, 0, Math.PI * 2);
        ctx.fillStyle = 'white'; ctx.fill();
    }

    if (start) {
        const x = start[1] * CELL_SIZE, y = start[0] * CELL_SIZE;
        ctx.fillStyle = MAZE_COLORS.start;
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
        ctx.fillStyle = 'white';
        ctx.font = `bold ${CELL_SIZE * 0.5}px Inter, sans-serif`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('S', x + CELL_SIZE / 2, y + CELL_SIZE / 2);
    }

    if (goal) {
        const x = goal[1] * CELL_SIZE, y = goal[0] * CELL_SIZE;
        ctx.fillStyle = MAZE_COLORS.goal;
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
        ctx.fillStyle = 'white';
        ctx.font = `bold ${CELL_SIZE * 0.5}px Inter, sans-serif`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('G', x + CELL_SIZE / 2, y + CELL_SIZE / 2);
    }
}

// -- Puzzle drawing -----------------------------------------------------------

function drawPuzzle(state, highlightCell = null) {
    const size  = canvas.width;
    const tileW = size / 3;
    const PAD   = 6;
    const R     = 14;

    ctx.fillStyle = '#0f0f14';
    ctx.fillRect(0, 0, size, size);

    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
            const val = state[r][c];
            const x = c * tileW + PAD, y = r * tileW + PAD;
            const w = tileW - PAD * 2, h = tileW - PAD * 2;

            if (val === 0) {
                ctx.fillStyle = '#1a1a26';
                roundRect(ctx, x, y, w, h, R);
                ctx.setLineDash([6, 4]);
                ctx.strokeStyle = '#2e2e42'; ctx.lineWidth = 2; ctx.stroke();
                ctx.setLineDash([]);
                continue;
            }

            const goalR = Math.floor((val - 1) / 3), goalC = (val - 1) % 3;
            const inPlace = (r === goalR && c === goalC);

            ctx.fillStyle = 'rgba(0,0,0,0.4)';
            roundRect(ctx, x + 3, y + 4, w, h, R);

            const color = PUZZLE_TILE_COLORS[val];
            ctx.fillStyle = inPlace ? lightenColor(color, 20) : color;
            roundRect(ctx, x, y, w, h, R);

            if (highlightCell && highlightCell[0] === r && highlightCell[1] === c) {
                ctx.fillStyle = 'rgba(255,255,255,0.25)';
                roundRect(ctx, x, y, w, h, R);
            }

            if (inPlace) {
                ctx.strokeStyle = 'rgba(255,255,255,0.4)'; ctx.lineWidth = 2.5;
                roundRect(ctx, x + 3, y + 3, w - 6, h - 6, R - 3); ctx.stroke();
            }

            ctx.fillStyle = 'rgba(255,255,255,0.95)';
            const fontSize = Math.floor(tileW * 0.38);
            ctx.font = `700 ${fontSize}px Inter, sans-serif`;
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(String(val), x + w / 2, y + h / 2);

            ctx.font = `400 ${Math.floor(fontSize * 0.42)}px Inter, sans-serif`;
            ctx.fillStyle = inPlace ? 'rgba(255,255,255,0.55)' : 'rgba(255,255,255,0.3)';
            ctx.fillText(`-${val}`, x + w - 18, y + h - 10);
        }
    }
}

// -- ARC drawing --------------------------------------------------------------

// drawARCGrid — draws a grid at (x0, y0). Labels are handled externally.
function drawARCGrid(grid, x0, y0, cellSize) {
    const rows = grid.length;
    const cols = grid[0].length;
    const gap  = 1;
    const gW   = cols * (cellSize + gap) - gap;
    const gH   = rows * (cellSize + gap) - gap;

    // Dark border background
    ctx.fillStyle = '#12121e';
    ctx.fillRect(x0 - 2, y0 - 2, gW + 4, gH + 4);

    // Cells
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const val = grid[r][c];
            ctx.fillStyle = ARC_COLORS[val] || '#111';
            ctx.fillRect(
                x0 + c * (cellSize + gap),
                y0 + r * (cellSize + gap),
                cellSize, cellSize
            );
        }
    }

    // Subtle outline
    ctx.strokeStyle = 'rgba(255,255,255,0.10)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x0 - 2, y0 - 2, gW + 4, gH + 4);
}

function drawARCTask(taskData, prediction = null) {
    const W = canvas.width;
    const H = canvas.height;

    ctx.fillStyle = '#0a0a10';
    ctx.fillRect(0, 0, W, H);

    const demos      = taskData.demos;
    const testInput  = taskData.test_input;
    const testOutput = taskData.test_output;
    const numDemos   = demos.length;

    // Max grid dimensions across all grids
    let maxR = 1, maxC = 1;
    demos.forEach(d => {
        maxR = Math.max(maxR, d.input.length,  d.output.length);
        maxC = Math.max(maxC, d.input[0]?.length  || 1, d.output[0]?.length || 1);
    });
    maxR = Math.max(maxR, testInput.length,  testOutput.length);
    maxC = Math.max(maxC, testInput[0]?.length || 1, testOutput[0]?.length || 1);

    // ── Layout constants ────────────────────────────────────────
    // Each vertical zone has its own explicit height:
    //   SEC_H  = section heading row (e.g. "DEMONSTRATIONS")
    //   LBL_H  = per-grid label row (e.g. "Demo 1 In")
    //   GRID   = the actual pixel grid
    //   GAP    = breathing room between rows / sections
    const MARGIN   = 14;
    const SEC_H    = 22;   // section header height
    const LBL_H    = 16;   // label height above each grid
    const GRID_GAP = 1;    // gap between cells inside a grid
    const ROW_GAP  = 18;   // vertical gap between consecutive demo rows
    const SEC_GAP  = 16;   // vertical gap between demo section and test section
    const ARROW_W  = 26;   // width of the → arrow zone
    const PAIR_GAP = 16;   // horizontal gap between demo pairs

    // Demo pair columns
    const perRow  = numDemos <= 2 ? numDemos : numDemos <= 4 ? 2 : 3;
    const numRows = Math.ceil(numDemos / perRow);

    // ── Cell size (fit everything into W × H) ───────────────────
    // Horizontal: W = 2*MARGIN + perRow*(2*maxC*cell + ARROW_W) + (perRow-1)*PAIR_GAP
    const availW   = W - 2*MARGIN - perRow*ARROW_W - Math.max(0, perRow-1)*PAIR_GAP;
    const cellByW  = Math.floor(availW / (2 * perRow * maxC));

    // Vertical: rows of demos + test row
    // Each demo row occupies: LBL_H + gridH
    // Between rows: ROW_GAP
    // Sections: SEC_H each, SEC_GAP between
    // Total fixed overhead (no cell-dependent):
    const fixedV   = 2*MARGIN + 2*SEC_H + (numRows + 1)*LBL_H + (numRows - 1)*ROW_GAP + SEC_GAP;
    // Total grid rows: (numRows demo rows + 1 test row) each of height = maxR * cell
    const cellByH  = Math.floor((H - fixedV) / ((numRows + 1) * maxR));

    const cellSize = Math.max(3, Math.min(cellByW, cellByH, 22));
    const gridW    = maxC * (cellSize + GRID_GAP) - GRID_GAP;
    const gridH    = maxR * (cellSize + GRID_GAP) - GRID_GAP;
    const pairW    = 2 * gridW + ARROW_W;

    // ── Drawing helpers ─────────────────────────────────────────
    function secHeader(text, x, y) {
        // Pill background
        ctx.fillStyle = 'rgba(100, 100, 160, 0.18)';
        const tw = ctx.measureText(text).width;
        ctx.beginPath();
        ctx.roundRect(x - 6, y - 1, tw + 12, 18, 4);
        ctx.fill();
        ctx.fillStyle = '#a0a8d8';
        ctx.font = 'bold 11px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.letterSpacing = '1.2px';
        ctx.fillText(text, x, y + 2);
        ctx.letterSpacing = '0px';
    }

    function gridLabel(text, x, y) {
        ctx.fillStyle = '#c8ccec';
        ctx.font      = '600 11px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(text, x, y);
    }

    function drawArrow(cx, cy) {
        ctx.fillStyle   = '#818cf8';
        ctx.font        = `bold ${Math.max(14, cellSize)}px Inter, sans-serif`;
        ctx.textAlign   = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('\u2192', cx, cy);
    }

    // ── DEMONSTRATIONS section ───────────────────────────────────
    let y = MARGIN;

    // Section header row
    secHeader('DEMONSTRATIONS', MARGIN, y);
    y += SEC_H;  // advance past section header — grids start after this

    const demoAreaTopY = y;

    for (let d = 0; d < numDemos; d++) {
        const col   = d % perRow;
        const row   = Math.floor(d / perRow);
        const xBase = MARGIN + col * (pairW + PAIR_GAP);

        // Each row occupies: LBL_H (label zone) + gridH (grid zone) + ROW_GAP
        const rowTopY  = demoAreaTopY + row * (LBL_H + gridH + ROW_GAP);
        const labelY   = rowTopY;            // label drawn here
        const gridY    = rowTopY + LBL_H;   // grid drawn below label

        // Labels (drawn FIRST at labelY, grids at gridY — no overlap)
        gridLabel(`Demo ${d + 1} In`, xBase, labelY);
        gridLabel('Out', xBase + gridW + ARROW_W, labelY);

        // Grids
        drawARCGrid(demos[d].input,  xBase,                  gridY, cellSize);
        drawArrow(xBase + gridW + ARROW_W / 2,               gridY + gridH / 2);
        drawARCGrid(demos[d].output, xBase + gridW + ARROW_W, gridY, cellSize);
    }

    // ── TEST section ─────────────────────────────────────────────
    const demoAreaH  = numRows * (LBL_H + gridH + ROW_GAP) - ROW_GAP;
    const testSecTopY = demoAreaTopY + demoAreaH + SEC_GAP;

    // Section header row
    secHeader('TEST', MARGIN, testSecTopY);
    const testLabelY = testSecTopY + SEC_H;   // label zone starts here
    const testGridY  = testLabelY  + LBL_H;  // grid starts below label

    gridLabel('Input', MARGIN, testLabelY);
    drawARCGrid(testInput, MARGIN, testGridY, cellSize);
    drawArrow(MARGIN + gridW + ARROW_W / 2, testGridY + gridH / 2);

    const predX = MARGIN + gridW + ARROW_W;

    if (prediction) {
        gridLabel('Prediction', predX, testLabelY);
        drawARCGrid(prediction, predX, testGridY, cellSize);

        // "vs" label
        const vsX = predX + gridW + ARROW_W / 2;
        ctx.fillStyle    = '#606080';
        ctx.font         = '600 10px Inter, sans-serif';
        ctx.textAlign    = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('vs', vsX, testGridY + gridH / 2);

        // Ground truth
        const gtX = predX + gridW + ARROW_W;
        if (gtX + gridW <= W - MARGIN) {
            gridLabel('Ground Truth', gtX, testLabelY);
            drawARCGrid(testOutput, gtX, testGridY, cellSize);
        }
    } else {
        // Placeholder box
        gridLabel('Prediction', predX, testLabelY);
        ctx.fillStyle = '#1a1a2e';
        ctx.beginPath();
        ctx.roundRect(predX, testGridY, gridW, gridH, 8);
        ctx.fill();
        ctx.strokeStyle = '#2e2e4a';
        ctx.lineWidth   = 1.5;
        ctx.stroke();

        ctx.fillStyle    = '#5050a0';
        ctx.font         = `bold ${Math.max(20, Math.min(Math.floor(gridH * 0.5), 38))}px Inter, sans-serif`;
        ctx.textAlign    = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('?', predX + gridW / 2, testGridY + gridH / 2);
    }
}

// -- Generate -----------------------------------------------------------------

async function generate() {
    if (isAnimating) return;
    if (currentTask === 'maze')        await generateMaze();
    else if (currentTask === 'puzzle') await generatePuzzle();
    else if (currentTask === 'arc')    await generateARC();
}

async function generateMaze() {
    setStatus('Generating...', 'solving');
    generateBtn.disabled = true;
    try {
        const res  = await fetch(`${API_BASE}/api/generate`);
        const data = await res.json();
        currentMaze  = data.grid;
        currentStart = data.start;
        currentGoal  = data.goal;
        canvasOverlay.classList.add('hidden');
        drawMaze(currentMaze, currentStart, currentGoal);
        optimalStepsEl.textContent = data.optimal_steps;
        agentStepsEl.textContent   = '\u2014';
        resultStatusEl.textContent = '\u2014'; resultStatusEl.style.color = '';
        setStatus('Ready to solve', 'ready');
        solveBtn.disabled = false;
    } catch (e) { setStatus('Error: ' + e.message, 'failed'); }
    generateBtn.disabled = false;
}

async function generatePuzzle() {
    setStatus('Generating puzzle...', 'solving');
    generateBtn.disabled = true;
    try {
        const res  = await fetch(`${API_BASE}/api/puzzle/generate`);
        const data = await res.json();
        currentPuzzle = data.state;
        canvasOverlay.classList.add('hidden');
        drawPuzzle(currentPuzzle);
        optimalStepsEl.textContent = data.optimal_steps;
        agentStepsEl.textContent   = '\u2014';
        resultStatusEl.textContent = '\u2014'; resultStatusEl.style.color = '';
        setStatus('Ready to solve', 'ready');
        solveBtn.disabled = false;
    } catch (e) { setStatus('Error: ' + e.message, 'failed'); }
    generateBtn.disabled = false;
}

async function generateARC() {
    setStatus('Loading ARC task...', 'solving');
    generateBtn.disabled = true;
    try {
        const res  = await fetch(`${API_BASE}/api/arc/generate`);
        const data = await res.json();
        if (data.error) {
            setStatus('Error: ' + data.error, 'failed');
            generateBtn.disabled = false;
            return;
        }
        currentARC = data;
        canvasOverlay.classList.add('hidden');
        drawARCTask(data);

        gridSizeEl.textContent     = `${data.input_size[0]}x${data.input_size[1]}`;
        optimalStepsEl.textContent = `${data.num_demos} demos`;
        agentStepsEl.textContent   = '\u2014';
        resultStatusEl.textContent = '\u2014'; resultStatusEl.style.color = '';
        setStatus(`Task: ${data.task_id}`, 'ready');
        solveBtn.disabled = false;
    } catch (e) { setStatus('Error: ' + e.message, 'failed'); }
    generateBtn.disabled = false;
}

// -- Solve --------------------------------------------------------------------

async function solve() {
    if (isAnimating) return;
    if (currentTask === 'maze'   && currentMaze)   await solveMaze();
    if (currentTask === 'puzzle' && currentPuzzle)  await solvePuzzle();
    if (currentTask === 'arc'    && currentARC)     await solveARC();
}

async function solveMaze() {
    isAnimating = true;
    solveBtn.disabled = true; generateBtn.disabled = true;
    setStatus('Agent solving...', 'solving');
    try {
        const res  = await fetch(`${API_BASE}/api/solve`, { method: 'POST' });
        const data = await res.json();
        if (data.error) {
            setStatus('Error: ' + data.error, 'failed');
            isAnimating = false; generateBtn.disabled = false; solveBtn.disabled = false;
            return;
        }
        const path = data.path;
        for (let i = 0; i < path.length; i++) {
            if (!isAnimating) break;
            drawMaze(currentMaze, currentStart, currentGoal, path.slice(0, i + 1));
            setStatus(`Solving... Step ${i}/${path.length - 1}`, 'solving');
            await sleep(animationSpeed);
        }
        agentStepsEl.textContent = data.steps;
        if (data.solved) {
            setStatus(`Solved in ${data.steps} steps`, 'success');
            resultStatusEl.textContent = 'Solved'; resultStatusEl.style.color = '#22c55e';
        } else {
            setStatus('Failed to solve', 'failed');
            resultStatusEl.textContent = 'Failed'; resultStatusEl.style.color = '#ef4444';
        }
    } catch (e) { setStatus('Error: ' + e.message, 'failed'); }
    isAnimating = false; generateBtn.disabled = false; solveBtn.disabled = false;
}

async function solvePuzzle() {
    isAnimating = true;
    solveBtn.disabled = true; generateBtn.disabled = true;
    setStatus('Agent solving...', 'solving');
    try {
        const res  = await fetch(`${API_BASE}/api/puzzle/solve`, { method: 'POST' });
        const data = await res.json();
        if (data.error) {
            setStatus('Error: ' + data.error, 'failed');
            isAnimating = false; generateBtn.disabled = false; solveBtn.disabled = false;
            return;
        }
        const states = data.states;
        for (let i = 0; i < states.length; i++) {
            if (!isAnimating) break;
            let highlight = null;
            if (i > 0) {
                const prev = states[i - 1], curr = states[i];
                for (let r = 0; r < 3; r++)
                    for (let c = 0; c < 3; c++)
                        if (prev[r][c] !== curr[r][c] && curr[r][c] !== 0) highlight = [r, c];
            }
            drawPuzzle(states[i], highlight);
            setStatus(`Step ${i}/${states.length - 1}`, 'solving');
            await sleep(animationSpeed);
        }
        agentStepsEl.textContent = data.steps;
        if (data.solved) {
            setStatus(`Solved in ${data.steps} steps`, 'success');
            resultStatusEl.textContent = 'Solved'; resultStatusEl.style.color = '#22c55e';
            await victoryFlash(states[states.length - 1]);
        } else {
            setStatus(`Failed (${data.steps} steps)`, 'failed');
            resultStatusEl.textContent = 'Failed'; resultStatusEl.style.color = '#ef4444';
        }
    } catch (e) { setStatus('Error: ' + e.message, 'failed'); }
    isAnimating = false; generateBtn.disabled = false; solveBtn.disabled = false;
}

async function solveARC() {
    isAnimating = true;
    solveBtn.disabled = true; generateBtn.disabled = true;
    setStatus('Model predicting...', 'solving');
    try {
        const res  = await fetch(`${API_BASE}/api/arc/solve`, { method: 'POST' });
        const data = await res.json();
        if (data.error) {
            setStatus('Error: ' + data.error, 'failed');
            isAnimating = false; generateBtn.disabled = false; solveBtn.disabled = false;
            return;
        }

        // Animate: reveal prediction cell by cell
        const pred = data.prediction;
        const rows = pred.length;
        const cols = pred[0].length;

        // Build partial prediction for animation
        const partial = pred.map(row => row.map(() => 0));
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                partial[r][c] = pred[r][c];
            }
            // Redraw with partial prediction every row
            drawARCTask(currentARC, partial);
            setStatus(`Predicting row ${r + 1}/${rows}...`, 'solving');
            await sleep(Math.max(20, animationSpeed / 2));
        }

        // Final draw with full prediction
        drawARCTask(currentARC, pred);

        const acc = Math.round(data.cell_accuracy * 100);
        agentStepsEl.textContent = `${acc}% cells`;
        if (data.grid_match) {
            setStatus(`Perfect match! (${acc}%)`, 'success');
            resultStatusEl.textContent = 'Matched'; resultStatusEl.style.color = '#22c55e';
        } else {
            setStatus(`Cell accuracy: ${acc}%`, 'failed');
            resultStatusEl.textContent = `${acc}%`; resultStatusEl.style.color = acc > 50 ? '#f59e0b' : '#ef4444';
        }
    } catch (e) { setStatus('Error: ' + e.message, 'failed'); }
    isAnimating = false; generateBtn.disabled = false; solveBtn.disabled = false;
}

async function victoryFlash(finalState) {
    for (let i = 0; i < 3; i++) {
        ctx.fillStyle = 'rgba(34,197,94,0.18)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        await sleep(120);
        drawPuzzle(finalState);
        await sleep(120);
    }
}

// -- Reset --------------------------------------------------------------------

function reset() {
    isAnimating   = false;
    currentMaze   = null;
    currentStart  = null;
    currentGoal   = null;
    currentPuzzle = null;
    currentARC    = null;

    drawBlankGrid();
    canvasOverlay.classList.remove('hidden');
    solveBtn.disabled = true;

    optimalStepsEl.textContent  = '\u2014';
    agentStepsEl.textContent    = '\u2014';
    resultStatusEl.textContent  = '\u2014';
    resultStatusEl.style.color  = '';
    setStatus('Ready', 'ready');
}

// -- Helpers ------------------------------------------------------------------

function setStatus(text, className) {
    statusText.textContent = text;
    statusText.className   = 'status-badge';
    if (className) statusText.classList.add(className);
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
    ctx.fill();
}

function lightenColor(hex, amount) {
    const num = parseInt(hex.slice(1), 16);
    const r = Math.min(255, (num >> 16) + amount);
    const g = Math.min(255, ((num >> 8) & 0xff) + amount);
    const b = Math.min(255, (num & 0xff) + amount);
    return `rgb(${r},${g},${b})`;
}
