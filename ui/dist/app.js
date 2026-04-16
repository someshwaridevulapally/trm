// RecursiveNet Task Solver UI

const API_BASE = '';

// DOM Elements
const canvas = document.getElementById('taskCanvas');
const ctx = canvas.getContext('2d');
const generateBtn = document.getElementById('generateBtn');
const solveBtn = document.getElementById('solveBtn');
const resetBtn = document.getElementById('resetBtn');
const canvasOverlay = document.getElementById('canvasOverlay');
const statusText = document.getElementById('statusText');
const optimalStepsEl = document.getElementById('optimalSteps');
const agentStepsEl = document.getElementById('agentSteps');
const resultStatusEl = document.getElementById('resultStatus');
const speedSlider = document.getElementById('speedSlider');
const speedValue = document.getElementById('speedValue');
const taskTitle = document.getElementById('taskTitle');
const taskDescription = document.getElementById('taskDescription');
const navTabs = document.querySelectorAll('.nav-tab');

// State
let currentTask = 'maze';
let currentMaze = null;
let currentStart = null;
let currentGoal = null;
let isAnimating = false;
let animationSpeed = 80;

// Task configurations
const TASKS = {
    maze: {
        title: 'Maze Solver',
        description: 'Navigate from start to goal using the trained agent',
        gridSize: 21,
        available: true
    },
    puzzle: {
        title: '8-Puzzle Solver',
        description: 'Slide tiles to reach the goal configuration',
        gridSize: 3,
        available: false
    },
    arc: {
        title: 'ARC-AGI Solver',
        description: 'Learn patterns from examples and predict outputs',
        gridSize: 30,
        available: false
    }
};

// Colors - high contrast for visibility
const COLORS = {
    wall: '#0d0d12',
    path: '#3d3d5c',
    start: '#10b981',
    goal: '#f97316',
    agent: '#8b5cf6',
    agentTrail: 'rgba(139, 92, 246, 0.35)'
};

// Cell size calculation
let GRID_SIZE = 21;
let CELL_SIZE = Math.floor(canvas.width / GRID_SIZE);

// Event Listeners
generateBtn.addEventListener('click', generate);
solveBtn.addEventListener('click', solve);
resetBtn.addEventListener('click', reset);

speedSlider.addEventListener('input', (e) => {
    animationSpeed = parseInt(e.target.value);
    speedValue.textContent = `${animationSpeed}ms`;
});

navTabs.forEach(tab => {
    tab.addEventListener('click', () => switchTask(tab.dataset.task));
});

// Initialize
init();

function init() {
    updateTaskUI();
    drawBlankGrid();
}

function switchTask(task) {
    if (task === currentTask || isAnimating) return;
    
    const taskConfig = TASKS[task];
    if (!taskConfig.available) {
        alert(`${taskConfig.title} is not yet implemented. Coming soon!`);
        return;
    }
    
    currentTask = task;
    navTabs.forEach(t => t.classList.remove('active'));
    document.querySelector(`[data-task="${task}"]`).classList.add('active');
    
    reset();
    updateTaskUI();
}

function updateTaskUI() {
    const config = TASKS[currentTask];
    taskTitle.textContent = config.title;
    taskDescription.textContent = config.description;
    GRID_SIZE = config.gridSize;
    CELL_SIZE = Math.floor(canvas.width / GRID_SIZE);
}

function drawBlankGrid() {
    ctx.fillStyle = COLORS.path;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    
    for (let i = 0; i <= GRID_SIZE; i++) {
        ctx.beginPath();
        ctx.moveTo(i * CELL_SIZE, 0);
        ctx.lineTo(i * CELL_SIZE, canvas.height);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(0, i * CELL_SIZE);
        ctx.lineTo(canvas.width, i * CELL_SIZE);
        ctx.stroke();
    }
}

function drawMaze(maze, start, goal, agentPath = []) {
    ctx.fillStyle = '#0f0f14';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw maze cells
    for (let r = 0; r < maze.length; r++) {
        for (let c = 0; c < maze[r].length; c++) {
            const x = c * CELL_SIZE;
            const y = r * CELL_SIZE;
            
            ctx.fillStyle = maze[r][c] === 1 ? COLORS.wall : COLORS.path;
            ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
        }
    }
    
    // Draw agent trail
    if (agentPath.length > 1) {
        ctx.fillStyle = COLORS.agentTrail;
        for (let i = 0; i < agentPath.length - 1; i++) {
            const step = agentPath[i];
            const x = step.col * CELL_SIZE;
            const y = step.row * CELL_SIZE;
            ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
        }
    }
    
    // Draw current agent position
    if (agentPath.length > 0) {
        const current = agentPath[agentPath.length - 1];
        const x = current.col * CELL_SIZE;
        const y = current.row * CELL_SIZE;
        
        ctx.fillStyle = COLORS.agent;
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
        
        // Agent marker
        ctx.beginPath();
        ctx.arc(x + CELL_SIZE / 2, y + CELL_SIZE / 2, CELL_SIZE / 3, 0, Math.PI * 2);
        ctx.fillStyle = 'white';
        ctx.fill();
    }
    
    // Draw start
    if (start) {
        const x = start[1] * CELL_SIZE;
        const y = start[0] * CELL_SIZE;
        
        ctx.fillStyle = COLORS.start;
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
        
        ctx.fillStyle = 'white';
        ctx.font = `bold ${CELL_SIZE * 0.5}px Inter, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('S', x + CELL_SIZE / 2, y + CELL_SIZE / 2);
    }
    
    // Draw goal
    if (goal) {
        const x = goal[1] * CELL_SIZE;
        const y = goal[0] * CELL_SIZE;
        
        ctx.fillStyle = COLORS.goal;
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
        
        ctx.fillStyle = 'white';
        ctx.font = `bold ${CELL_SIZE * 0.5}px Inter, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('G', x + CELL_SIZE / 2, y + CELL_SIZE / 2);
    }
}

async function generate() {
    if (isAnimating) return;
    
    if (currentTask === 'maze') {
        await generateMaze();
    }
}

async function generateMaze() {
    setStatus('Generating...', 'solving');
    generateBtn.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/api/generate`);
        const data = await response.json();
        
        currentMaze = data.grid;
        currentStart = data.start;
        currentGoal = data.goal;
        
        canvasOverlay.classList.add('hidden');
        drawMaze(currentMaze, currentStart, currentGoal);
        
        optimalStepsEl.textContent = data.optimal_steps;
        agentStepsEl.textContent = '—';
        resultStatusEl.textContent = '—';
        resultStatusEl.style.color = '';
        
        setStatus('Ready to solve', 'ready');
        solveBtn.disabled = false;
    } catch (error) {
        setStatus('Error: ' + error.message, 'failed');
    }
    
    generateBtn.disabled = false;
}

async function solve() {
    if (isAnimating || !currentMaze) return;
    
    if (currentTask === 'maze') {
        await solveMaze();
    }
}

async function solveMaze() {
    isAnimating = true;
    solveBtn.disabled = true;
    generateBtn.disabled = true;
    
    setStatus('Agent solving...', 'solving');
    
    try {
        const response = await fetch(`${API_BASE}/api/solve`, { method: 'POST' });
        const data = await response.json();
        
        if (data.error) {
            setStatus('Error: ' + data.error, 'failed');
            isAnimating = false;
            generateBtn.disabled = false;
            solveBtn.disabled = false;
            return;
        }
        
        // Animate the path
        const path = data.path;
        for (let i = 0; i < path.length; i++) {
            if (!isAnimating) break;
            
            drawMaze(currentMaze, currentStart, currentGoal, path.slice(0, i + 1));
            setStatus(`Solving... Step ${i}/${path.length - 1}`, 'solving');
            
            await sleep(animationSpeed);
        }
        
        // Final state
        agentStepsEl.textContent = data.steps;
        
        if (data.solved) {
            setStatus(`Solved in ${data.steps} steps`, 'success');
            resultStatusEl.textContent = 'Solved';
            resultStatusEl.style.color = '#22c55e';
        } else {
            setStatus('Failed to solve', 'failed');
            resultStatusEl.textContent = 'Failed';
            resultStatusEl.style.color = '#ef4444';
        }
        
    } catch (error) {
        setStatus('Error: ' + error.message, 'failed');
    }
    
    isAnimating = false;
    generateBtn.disabled = false;
    solveBtn.disabled = false;
}

function reset() {
    isAnimating = false;
    currentMaze = null;
    currentStart = null;
    currentGoal = null;
    
    drawBlankGrid();
    canvasOverlay.classList.remove('hidden');
    
    solveBtn.disabled = true;
    
    optimalStepsEl.textContent = '—';
    agentStepsEl.textContent = '—';
    resultStatusEl.textContent = '—';
    resultStatusEl.style.color = '';
    
    setStatus('Ready', 'ready');
}

function setStatus(text, className) {
    statusText.textContent = text;
    statusText.className = 'status-badge';
    if (className) {
        statusText.classList.add(className);
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
