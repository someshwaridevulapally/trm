// Maze Solver UI - JavaScript Application

const API_BASE = '';

// DOM Elements
const canvas = document.getElementById('mazeCanvas');
const ctx = canvas.getContext('2d');
const generateBtn = document.getElementById('generateBtn');
const solveBtn = document.getElementById('solveBtn');
const solveBfsBtn = document.getElementById('solveBfsBtn');
const resetBtn = document.getElementById('resetBtn');
const mazeOverlay = document.getElementById('mazeOverlay');
const statusText = document.getElementById('statusText');
const optimalStepsEl = document.getElementById('optimalSteps');
const agentStepsEl = document.getElementById('agentSteps');
const resultStatusEl = document.getElementById('resultStatus');
const speedSlider = document.getElementById('speedSlider');
const speedValue = document.getElementById('speedValue');

// State
let currentMaze = null;
let currentStart = null;
let currentGoal = null;
let isAnimating = false;
let animationSpeed = 100;

// Colors
const COLORS = {
    wall: '#1a1a2e',
    path: '#2a2a4e',
    start: '#38ef7d',
    goal: '#f5576c',
    agent: '#00d9ff',
    agentTrail: 'rgba(0, 217, 255, 0.3)',
    bfsPath: '#ffd700',
    bfsTrail: 'rgba(255, 215, 0, 0.3)'
};

// Cell size calculation
const GRID_SIZE = 21;
const CELL_SIZE = Math.floor(canvas.width / GRID_SIZE);

// Event Listeners
generateBtn.addEventListener('click', generateMaze);
solveBtn.addEventListener('click', () => solveMaze('ai'));
solveBfsBtn.addEventListener('click', () => solveMaze('bfs'));
resetBtn.addEventListener('click', reset);
speedSlider.addEventListener('input', (e) => {
    animationSpeed = parseInt(e.target.value);
    speedValue.textContent = `${animationSpeed}ms`;
});

// Initialize
drawBlankGrid();

function drawBlankGrid() {
    ctx.fillStyle = COLORS.path;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 0.5;
    
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

function drawMaze(maze, start, goal, agentPath = [], pathColor = COLORS.agent, trailColor = COLORS.agentTrail) {
    // Clear canvas
    ctx.fillStyle = '#0a0a15';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw maze cells
    for (let r = 0; r < maze.length; r++) {
        for (let c = 0; c < maze[r].length; c++) {
            const x = c * CELL_SIZE;
            const y = r * CELL_SIZE;
            
            if (maze[r][c] === 1) {
                ctx.fillStyle = COLORS.wall;
            } else {
                ctx.fillStyle = COLORS.path;
            }
            ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
        }
    }
    
    // Draw agent trail
    if (agentPath.length > 1) {
        ctx.fillStyle = trailColor;
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
        ctx.fillStyle = pathColor;
        const x = current.col * CELL_SIZE;
        const y = current.row * CELL_SIZE;
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
        
        // Draw agent circle
        ctx.beginPath();
        ctx.arc(x + CELL_SIZE / 2, y + CELL_SIZE / 2, CELL_SIZE / 3, 0, Math.PI * 2);
        ctx.fillStyle = 'white';
        ctx.fill();
    }
    
    // Draw start
    if (start) {
        ctx.fillStyle = COLORS.start;
        const x = start[1] * CELL_SIZE;
        const y = start[0] * CELL_SIZE;
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
        
        // Draw S
        ctx.fillStyle = 'white';
        ctx.font = `bold ${CELL_SIZE * 0.6}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('S', x + CELL_SIZE / 2, y + CELL_SIZE / 2);
    }
    
    // Draw goal
    if (goal) {
        ctx.fillStyle = COLORS.goal;
        const x = goal[1] * CELL_SIZE;
        const y = goal[0] * CELL_SIZE;
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
        
        // Draw G
        ctx.fillStyle = 'white';
        ctx.font = `bold ${CELL_SIZE * 0.6}px Arial`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('G', x + CELL_SIZE / 2, y + CELL_SIZE / 2);
    }
}

async function generateMaze() {
    if (isAnimating) return;
    
    setStatus('Generating maze...', 'solving');
    generateBtn.disabled = true;
    
    try {
        const response = await fetch(`${API_BASE}/api/generate`);
        const data = await response.json();
        
        currentMaze = data.grid;
        currentStart = data.start;
        currentGoal = data.goal;
        
        mazeOverlay.classList.add('hidden');
        drawMaze(currentMaze, currentStart, currentGoal);
        
        optimalStepsEl.textContent = data.optimal_steps;
        agentStepsEl.textContent = '-';
        resultStatusEl.textContent = '-';
        
        setStatus('Maze generated! Click Solve to start.', '');
        solveBtn.disabled = false;
        solveBfsBtn.disabled = false;
    } catch (error) {
        setStatus('Error generating maze: ' + error.message, 'failed');
    }
    
    generateBtn.disabled = false;
}

async function solveMaze(method = 'ai') {
    if (isAnimating || !currentMaze) return;
    
    isAnimating = true;
    solveBtn.disabled = true;
    solveBfsBtn.disabled = true;
    generateBtn.disabled = true;
    
    const isAI = method === 'ai';
    const endpoint = isAI ? '/api/solve' : '/api/solve_bfs';
    const pathColor = isAI ? COLORS.agent : COLORS.bfsPath;
    const trailColor = isAI ? COLORS.agentTrail : COLORS.bfsTrail;
    
    setStatus(isAI ? '🤖 AI Agent solving...' : '📐 BFS solving...', 'solving');
    
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, { method: 'POST' });
        const data = await response.json();
        
        if (data.error) {
            setStatus('Error: ' + data.error, 'failed');
            isAnimating = false;
            generateBtn.disabled = false;
            solveBtn.disabled = false;
            solveBfsBtn.disabled = false;
            return;
        }
        
        // Animate the path
        const path = data.path;
        for (let i = 0; i < path.length; i++) {
            if (!isAnimating) break;
            
            drawMaze(currentMaze, currentStart, currentGoal, path.slice(0, i + 1), pathColor, trailColor);
            setStatus(`${isAI ? '🤖 AI Agent' : '📐 BFS'}: Step ${i}/${path.length - 1}`, 'solving');
            
            await sleep(animationSpeed);
        }
        
        // Final state
        agentStepsEl.textContent = data.steps;
        
        if (data.solved) {
            setStatus(`✓ ${isAI ? 'AI Agent' : 'BFS'} solved in ${data.steps} steps!`, 'success');
            resultStatusEl.textContent = '✓ Solved';
            resultStatusEl.style.color = '#38ef7d';
        } else {
            setStatus(`✗ ${isAI ? 'AI Agent' : 'BFS'} failed to solve`, 'failed');
            resultStatusEl.textContent = '✗ Failed';
            resultStatusEl.style.color = '#f5576c';
        }
        
    } catch (error) {
        setStatus('Error solving: ' + error.message, 'failed');
    }
    
    isAnimating = false;
    generateBtn.disabled = false;
    solveBtn.disabled = false;
    solveBfsBtn.disabled = false;
}

function reset() {
    isAnimating = false;
    currentMaze = null;
    currentStart = null;
    currentGoal = null;
    
    drawBlankGrid();
    mazeOverlay.classList.remove('hidden');
    
    solveBtn.disabled = true;
    solveBfsBtn.disabled = true;
    
    optimalStepsEl.textContent = '-';
    agentStepsEl.textContent = '-';
    resultStatusEl.textContent = '-';
    resultStatusEl.style.color = '';
    
    setStatus('Ready', '');
}

function setStatus(text, className) {
    statusText.textContent = text;
    statusText.className = 'status-text';
    if (className) {
        statusText.classList.add(className);
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
