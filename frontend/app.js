// ═══════════════════════════════════════════════════════
//  RAG Document Analyst — Frontend App Logic
//  Communicates with FastAPI backend via fetch()
// ═══════════════════════════════════════════════════════

// ── Configuration ────────────────────────────────────────
// Update BACKEND_URL when deploying:
//   • Local dev  → "http://localhost:8000"
//   • Render     → "https://your-service-name.onrender.com"
const BACKEND_URL = "http://localhost:8000";

// ── State ────────────────────────────────────────────────
let sessionId = null;
let isProcessing = false;
let isChatting = false;

// ── DOM References ───────────────────────────────────────
const dropZone         = document.getElementById("drop-zone");
const fileInput        = document.getElementById("file-input");
const filePreview      = document.getElementById("file-preview");
const fileName         = document.getElementById("file-name");
const fileSize         = document.getElementById("file-size");
const removeFileBtn    = document.getElementById("remove-file-btn");
const processBtn       = document.getElementById("process-btn");
const statusPanel      = document.getElementById("status-panel");
const sessionInfo      = document.getElementById("session-info");
const statSession      = document.getElementById("stat-session");
const statChunks       = document.getElementById("stat-chunks");
const messagesContainer= document.getElementById("messages-container");
const emptyState       = document.getElementById("empty-state");
const typingIndicator  = document.getElementById("typing-indicator");
const messageInput     = document.getElementById("message-input");
const sendBtn          = document.getElementById("send-btn");
const clearChatBtn     = document.getElementById("clear-chat-btn");
const examplePills     = document.querySelectorAll(".example-pill");

// ── File Handling ────────────────────────────────────────

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(2) + " MB";
}

function setSelectedFile(file) {
  if (!file) return;
  const allowed = ["pdf", "docx", "xlsx", "xls"];
  const ext = file.name.split(".").pop().toLowerCase();

  if (!allowed.includes(ext)) {
    showStatus("❌ Invalid file type. Please upload PDF, DOCX, or Excel.", "error");
    fileInput.value = "";
    return;
  }

  fileName.textContent = file.name;
  fileSize.textContent = formatBytes(file.size);
  filePreview.classList.remove("hidden");
  processBtn.disabled = false;
  processBtn.setAttribute("aria-disabled", "false");
  dropZone.setAttribute("aria-label", `Selected: ${file.name}. Click or drag to replace.`);
  clearStatus();
}

function clearFileSelection() {
  fileInput.value = "";
  filePreview.classList.add("hidden");
  processBtn.disabled = true;
  processBtn.setAttribute("aria-disabled", "true");
  dropZone.setAttribute("aria-label", "Upload document — click or drag and drop a PDF, DOCX, or Excel file");
}

// Drag and drop
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer?.files?.[0];
  if (file) setSelectedFile(file);
});

// Click / keyboard on drop zone
dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

fileInput.addEventListener("change", () => {
  if (fileInput.files?.[0]) setSelectedFile(fileInput.files[0]);
});

removeFileBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  clearFileSelection();
  clearStatus();
});

// ── Status Panel ─────────────────────────────────────────

function showStatus(message, type = "loading") {
  statusPanel.textContent = message;
  statusPanel.className = `status-panel ${type}`;
  statusPanel.classList.remove("hidden");
}
function clearStatus() {
  statusPanel.className = "status-panel hidden";
  statusPanel.textContent = "";
}

// ── Upload & Process ─────────────────────────────────────

processBtn.addEventListener("click", uploadDocument);

async function uploadDocument() {
  const file = fileInput.files?.[0];
  if (!file || isProcessing) return;

  isProcessing = true;
  processBtn.disabled = true;
  document.querySelector(".btn-text").textContent = "Processing…";
  document.querySelector(".btn-spinner").classList.remove("hidden");
  showStatus("⏳ Uploading and processing your document…", "loading");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${BACKEND_URL}/upload`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Unknown error." }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    sessionId = data.session_id;

    showStatus(data.message, "success");

    // Show session info
    statSession.textContent = sessionId.slice(0, 8) + "…";
    statChunks.textContent  = data.chunks + " chunks";
    sessionInfo.classList.remove("hidden");

    // Enable chat
    messageInput.disabled = false;
    messageInput.setAttribute("aria-disabled", "false");
    sendBtn.disabled = false;
    sendBtn.setAttribute("aria-disabled", "false");
    messageInput.focus();

  } catch (err) {
    showStatus(`❌ Error: ${err.message}`, "error");
    processBtn.disabled = false;
  } finally {
    isProcessing = false;
    document.querySelector(".btn-text").textContent = "Process Document";
    document.querySelector(".btn-spinner").classList.add("hidden");
  }
}

// ── Chat ─────────────────────────────────────────────────

// Auto-resize textarea
messageInput.addEventListener("input", () => {
  messageInput.style.height = "auto";
  messageInput.style.height = Math.min(messageInput.scrollHeight, 160) + "px";
});

// Send on Enter (Shift+Enter = newline)
messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener("click", sendMessage);

// Example pills
examplePills.forEach((pill) => {
  pill.addEventListener("click", () => {
    if (!sessionId) {
      showStatus("⚠️ Please upload a document first.", "error");
      return;
    }
    messageInput.value = pill.dataset.question;
    messageInput.dispatchEvent(new Event("input")); // trigger resize
    sendMessage();
  });
});

async function sendMessage() {
  if (!sessionId) {
    showStatus("⚠️ Please upload a document before chatting.", "error");
    return;
  }
  const question = messageInput.value.trim();
  if (!question || isChatting) return;

  isChatting = true;
  messageInput.value = "";
  messageInput.style.height = "auto";
  sendBtn.disabled = true;

  // Remove empty state
  emptyState?.remove();

  // Add user bubble
  appendMessage("user", question);

  // Show typing indicator
  typingIndicator.classList.remove("hidden");
  scrollToBottom();

  try {
    const res = await fetch(`${BACKEND_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, question }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Unknown error." }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    typingIndicator.classList.add("hidden");
    appendMessage("ai", data.answer);

  } catch (err) {
    typingIndicator.classList.add("hidden");
    appendMessage("ai", `❌ Error: ${err.message}`, true);
  } finally {
    isChatting = false;
    sendBtn.disabled = false;
    messageInput.focus();
    scrollToBottom();
  }
}

// ── Message Rendering ─────────────────────────────────────

function appendMessage(role, text, isError = false) {
  const wrapper = document.createElement("div");
  wrapper.classList.add("message", `message--${role}`);
  wrapper.setAttribute("role", "article");
  wrapper.setAttribute("aria-label", `${role === "user" ? "You" : "AI"}: ${text.slice(0, 80)}`);

  const avatar = document.createElement("div");
  avatar.classList.add("message-avatar");
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = role === "user" ? "YOU" : "AI";

  const bubble = document.createElement("div");
  bubble.classList.add("message-bubble");
  if (isError) bubble.style.borderColor = "hsla(0, 70%, 60%, 0.4)";

  // Simple markdown-like rendering for AI responses
  if (role === "ai") {
    bubble.innerHTML = renderMarkdown(text);
  } else {
    bubble.textContent = text;
  }

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  messagesContainer.appendChild(wrapper);
  scrollToBottom();
}

/** Minimal markdown renderer for AI output */
function renderMarkdown(text) {
  return text
    // Bold
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    // Italic
    .replace(/\*(.*?)\*/g, "<em>$1</em>")
    // Inline code
    .replace(/`([^`]+)`/g, "<code style='background:hsla(260,30%,20%,0.7);padding:2px 6px;border-radius:4px;font-family:monospace;font-size:0.85em;'>$1</code>")
    // Bullet lists
    .replace(/^[-•]\s+(.+)/gm, "<li>$1</li>")
    .replace(/(<li>.*<\/li>)/gs, "<ul style='padding-left:1.2em;margin:6px 0;'>$1</ul>")
    // Numbered lists
    .replace(/^\d+\.\s+(.+)/gm, "<li>$1</li>")
    // Line breaks
    .replace(/\n{2,}/g, "</p><p style='margin-top:8px;'>")
    .replace(/\n/g, "<br>")
    // Wrap in paragraph
    .replace(/^(.)/s, "<p>$1")
    .replace(/(.)$/s, "$1</p>");
}

function scrollToBottom() {
  messagesContainer.scrollTo({ top: messagesContainer.scrollHeight, behavior: "smooth" });
}

// ── Clear Chat ────────────────────────────────────────────

clearChatBtn.addEventListener("click", () => {
  // Remove all messages except empty state
  const messages = messagesContainer.querySelectorAll(".message");
  messages.forEach(m => m.remove());

  // Re-add empty state if it was removed
  if (!document.getElementById("empty-state")) {
    const es = document.createElement("div");
    es.id = "empty-state";
    es.className = "empty-state";
    es.innerHTML = `
      <div class="empty-state-icon" aria-hidden="true">
        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
      </div>
      <h2 class="empty-state-title">Chat Cleared</h2>
      <p class="empty-state-desc">Your document is still loaded. Ask a new question below.</p>
    `;
    messagesContainer.appendChild(es);
  }
});

// ── Health Check on Load ──────────────────────────────────

(async function checkHealth() {
  try {
    const res = await fetch(`${BACKEND_URL}/health`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error("Not OK");
    console.log("✅ Backend connected");
  } catch {
    console.warn("⚠️ Backend not reachable at", BACKEND_URL, "— start it with: uvicorn main:app --reload");
  }
})();
