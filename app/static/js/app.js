const input = document.getElementById("inputText");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const copyBtn = document.getElementById("copyBtn");
const wordCount = document.getElementById("wordCount");
const emptyState = document.getElementById("emptyState");
const resultContent = document.getElementById("resultContent");
const highlightedOutput = document.getElementById("highlightedOutput");
const entityCount = document.getElementById("entityCount");
const latency = document.getElementById("latency");
const apiCalls = document.getElementById("apiCalls");
const cacheHits = document.getElementById("cacheHits");
const toast = document.getElementById("toast");

let lastResult = null;

const urduTokenRegex = /[\u0600-\u06FF]+|[0-9۰-۹]+/g;

function countWords(text) {
    return (text.match(urduTokenRegex) || []).length;
}

function updateWordCount() {
    const count = countWords(input.value);
    wordCount.textContent = count;

    if (count > window.APP_CONFIG.maxWords) {
        wordCount.style.color = "#dc7c93";
    } else {
        wordCount.style.color = "";
    }
}

function showToast(message, isError = false) {
    toast.textContent = message;
    toast.classList.toggle("error", isError);
    toast.classList.add("show");

    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(() => {
        toast.classList.remove("show");
    }, 3200);
}

function escapeHtml(text) {
    return text
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
}

function buildHighlightedText(text, entities) {
    const valid = entities
        .filter(e => Number.isInteger(e.start) && Number.isInteger(e.end))
        .sort((a, b) => a.start - b.start);

    if (!valid.length) {
        return escapeHtml(text);
    }

    let html = "";
    let cursor = 0;

    for (const entity of valid) {
        if (entity.start < cursor) continue;

        html += escapeHtml(text.slice(cursor, entity.start));

        const safeEntity = escapeHtml(text.slice(entity.start, entity.end));
        html += `<span class="entity-highlight" data-label="${entity.label}">${safeEntity}</span>`;

        cursor = entity.end;
    }

    html += escapeHtml(text.slice(cursor));
    return html;
}

function setLoading(loading) {
    analyzeBtn.disabled = loading;
    analyzeBtn.classList.toggle("loading", loading);
    analyzeBtn.querySelector(".button-label").textContent =
        loading ? "Analyzing Urdu…" : "Analyze Entities";
}

function resetResult() {
    lastResult = null;
    highlightedOutput.innerHTML = "";
    emptyState.classList.remove("hidden");
    resultContent.classList.add("hidden");
    copyBtn.disabled = true;

    entityCount.textContent = "0";
    latency.textContent = "—";
    apiCalls.textContent = "—";
    cacheHits.textContent = "—";
}

async function analyze() {
    const text = input.value.trim();

    if (!text) {
        showToast("Please enter an Urdu sentence.", true);
        input.focus();
        return;
    }

    const words = countWords(text);
    if (words > window.APP_CONFIG.maxWords) {
        showToast(
            `This demo supports up to ${window.APP_CONFIG.maxWords} Urdu words per request.`,
            true
        );
        return;
    }

    setLoading(true);

    try {
        const response = await fetch("/api/ner", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text })
        });

        const payload = await response.json();

        if (!response.ok) {
            const message = payload.detail || "Unable to analyze this text.";
            throw new Error(message);
        }

        lastResult = payload;

        highlightedOutput.innerHTML =
            buildHighlightedText(payload.input, payload.entities);

        entityCount.textContent = payload.entity_count;
        latency.textContent = `${payload.latency_ms} ms`;
        apiCalls.textContent = payload.api_calls;
        cacheHits.textContent = payload.cache_hits;

        emptyState.classList.add("hidden");
        resultContent.classList.remove("hidden");
        copyBtn.disabled = false;

        if (payload.api_calls === 0) {
            showToast("Result loaded from cache.");
        } else {
            showToast(`${payload.entity_count} entities identified.`);
        }

    } catch (error) {
        showToast(error.message || "Something went wrong.", true);
    } finally {
        setLoading(false);
    }
}

input.addEventListener("input", updateWordCount);

input.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        analyze();
    }
});

analyzeBtn.addEventListener("click", analyze);

clearBtn.addEventListener("click", () => {
    input.value = "";
    updateWordCount();
    resetResult();
    input.focus();
});

copyBtn.addEventListener("click", async () => {
    if (!lastResult) return;

    const lines = lastResult.entities.map(
        e => `${e.text} → ${e.label}`
    );

    const copyText = lines.length
        ? lines.join("\n")
        : "No supported entities detected.";

    try {
        await navigator.clipboard.writeText(copyText);
        showToast("Entity result copied.");
    } catch {
        showToast("Could not access clipboard.", true);
    }
});

document.querySelectorAll(".example-card").forEach(card => {
    card.addEventListener("click", () => {
        input.value = card.dataset.example || "";
        updateWordCount();
        resetResult();

        document.querySelector(".workspace").scrollIntoView({
            behavior: "smooth",
            block: "center"
        });

        window.setTimeout(() => input.focus(), 500);
    });
});

updateWordCount();
