const API_BASE = "http://127.0.0.1:8000";

const chatBox = document.getElementById('chat-box');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const userIdInput = document.getElementById('user-id');
const loadMemoryBtn = document.getElementById('load-memory-btn');
const memoryDisplay = document.getElementById('memory-display');

function getUserId() {
    return userIdInput.value.trim() || 'test_user';
}

function appendMessage(role, content) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', role);
    
    const bubble = document.createElement('div');
    bubble.classList.add('bubble');
    
    if (content === 'typing...') {
        bubble.innerHTML = `<div class="typing-indicator"><span></span><span></span><span></span></div>`;
        msgDiv.id = "typing-msg";
    } else {
        bubble.innerText = content;
    }
    
    msgDiv.appendChild(bubble);
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function removeTypingIndicator() {
    const typingMsg = document.getElementById('typing-msg');
    if (typingMsg) typingMsg.remove();
}

async function fetchMemory() {
    const defaultText = loadMemoryBtn.innerText;
    try {
        loadMemoryBtn.innerText = "Loading...";
        const res = await fetch(`${API_BASE}/v1/memory/${getUserId()}`);
        if (!res.ok) throw new Error('Failed to load memory');
        const data = await res.json();
        
        memoryDisplay.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
    } catch (err) {
        memoryDisplay.innerHTML = `<p style="color: #ef4444;">Error: ${err.message}</p>`;
    } finally {
        loadMemoryBtn.innerText = defaultText;
    }
}

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;
    
    appendMessage('user', text);
    chatInput.value = '';
    
    // 创建一个空白的 AI 会话气泡，等待分块流式写入
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', 'ai');
    const bubble = document.createElement('div');
    bubble.classList.add('bubble');
    msgDiv.appendChild(bubble);
    chatBox.appendChild(msgDiv);
    
    try {
        const res = await fetch(`${API_BASE}/v1/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: getUserId(), message: text })
        });
        
        if (!res.ok) throw new Error('API Error');
        
        // 核心 Streaming 解码逻辑
        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let done = false;
        
        while (!done) {
            const { value, done: readerDone } = await reader.read();
            done = readerDone;
            if (value) {
                const chunk = decoder.decode(value, { stream: true });
                bubble.innerText += chunk;
                // 确保视图跟随打字机下滑
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        }
        
        // 当流结束后，后台 reflect 节点大概率也结束落盘了，静默获取最新记忆
        fetchMemory();
    } catch (err) {
        bubble.innerText = 'Error sending message. Is the FastAPI server running?';
    }
}

sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

async function fetchHistory() {
    try {
        const res = await fetch(`${API_BASE}/v1/history/${getUserId()}`);
        if (!res.ok) throw new Error('Failed to load history');
        const historyData = await res.json();
        
        chatBox.innerHTML = `
            <div class="message system">
                <div class="bubble">Welcome! Set your User ID on the left and start chatting to evolve the digital twin's memory.</div>
            </div>
        `;
        
        for (const item of historyData) {
            const lines = item.content.split('\nAgent: ');
            if (lines.length === 2) {
                const userMsg = lines[0].replace('User: ', '');
                const aiMsg = lines[1];
                appendMessage('user', userMsg);
                appendMessage('ai', aiMsg);
            }
        }
        chatBox.scrollTop = chatBox.scrollHeight;
    } catch (err) {
        console.error("Error fetching history:", err);
    }
}

loadMemoryBtn.addEventListener('click', async () => {
    await fetchMemory();
    await fetchHistory();
});

// Initial memory & history load
(async () => {
    await fetchMemory();
    await fetchHistory();
})();
