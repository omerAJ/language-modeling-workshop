var LIVE_DEMO_ENDPOINT = 'https://api.openai.com/v1/chat/completions';
var LIVE_DEMO_MODEL = 'gpt-3.5-turbo-0125';

function getLiveDemoElements() {
  var root = document.getElementById('slide-32b');
  if (!root) return {};
  return {
    root: root,
    apiKeyInput: document.getElementById('liveDemoApiKey'),
    promptInput: document.getElementById('liveDemoInput'),
    messages: document.getElementById('liveDemoMessages'),
    status: document.getElementById('liveDemoStatus'),
    sendButton: root.querySelector('[data-demo-send]'),
    clearButton: root.querySelector('[data-demo-clear]'),
    presetButtons: Array.from(root.querySelectorAll('[data-demo-preset]'))
  };
}

function clearLiveDemoPresetTimer() {
  if (!liveDemoState.presetTimer) return;
  clearTrackedTimeout(liveDemoState.presetTimer);
  liveDemoState.presetTimer = null;
}

function ensureLiveDemoEmptyState() {
  var elements = getLiveDemoElements();
  if (!elements.messages) return;
  if (elements.messages.querySelector('.live-demo-message') || elements.messages.querySelector('.live-demo-empty')) return;

  var empty = document.createElement('div');
  empty.className = 'live-demo-empty';
  empty.textContent = 'Tap a probe or type your own prompt. The transcript stays here until you clear context.';
  elements.messages.appendChild(empty);
}

function syncLiveDemoScroll() {
  var elements = getLiveDemoElements();
  if (!elements.messages) return;
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

function setLiveDemoStatus(message, tone) {
  var elements = getLiveDemoElements();
  if (!elements.status) return;
  elements.status.textContent = message || '';
  elements.status.dataset.tone = tone || 'info';
  scheduleActiveSlideFit({ reason: 'live-demo-status', dispatchResize: false });
}

function setLiveDemoBusy(isBusy) {
  var elements = getLiveDemoElements();
  if (!elements.root) return;

  liveDemoState.isLoading = !!isBusy;
  elements.root.classList.toggle('live-demo-busy', !!isBusy);

  if (elements.sendButton) elements.sendButton.disabled = !!isBusy;
  if (elements.promptInput) elements.promptInput.disabled = !!isBusy;
  if (elements.apiKeyInput) elements.apiKeyInput.readOnly = !!isBusy;
  if (elements.clearButton) elements.clearButton.textContent = isBusy ? 'Stop & clear' : 'Clear context';
  elements.presetButtons.forEach(function(button) {
    button.disabled = !!isBusy;
  });
}

function createLiveDemoMessage(role, text, options) {
  var opts = options || {};
  var bubble = document.createElement('div');
  bubble.className = 'live-demo-message ' + role;
  if (opts.loading) bubble.classList.add('loading');
  if (role === 'user' || role === 'assistant') bubble.dataset.role = role;

  var label = document.createElement('div');
  label.className = 'live-demo-message-label';
  label.textContent = role === 'user' ? 'You' : 'Model';

  var body = document.createElement('div');
  body.className = 'live-demo-message-body';
  body.textContent = text;

  bubble.appendChild(label);
  bubble.appendChild(body);
  return bubble;
}

function appendLiveDemoMessage(role, text, options) {
  var elements = getLiveDemoElements();
  if (!elements.messages) return null;

  var empty = elements.messages.querySelector('.live-demo-empty');
  if (empty) empty.remove();

  var bubble = createLiveDemoMessage(role, text, options);
  elements.messages.appendChild(bubble);
  syncLiveDemoScroll();
  scheduleActiveSlideFit({ reason: 'live-demo-message', dispatchResize: false });
  return bubble;
}

function getLiveDemoConversationMessages() {
  var elements = getLiveDemoElements();
  if (!elements.messages) return [];

  return Array.from(elements.messages.querySelectorAll('.live-demo-message[data-role]')).map(function(node) {
    var body = node.querySelector('.live-demo-message-body');
    return {
      role: node.dataset.role,
      content: body ? body.textContent : ''
    };
  }).filter(function(message) {
    return message.content && !message.content.trim().startsWith('Thinking');
  });
}

function getLiveDemoResponseText(payload) {
  if (!payload || !payload.choices || !payload.choices.length) return '';

  var choice = payload.choices[0] || {};
  var message = choice.message || {};
  var content = message.content;

  if (typeof content === 'string') return content.trim();
  if (Array.isArray(content)) {
    return content.map(function(part) {
      if (!part) return '';
      if (typeof part === 'string') return part;
      if (typeof part.text === 'string') return part.text;
      return '';
    }).join('').trim();
  }
  return '';
}

function abortLiveDemoRequest(suppressAbortStatus) {
  if (!liveDemoState.abortController) return;
  liveDemoState.suppressAbortStatus = !!suppressAbortStatus;
  liveDemoState.abortController.abort();
  liveDemoState.abortController = null;
}

function clearLiveFailureDemo() {
  var elements = getLiveDemoElements();
  clearLiveDemoPresetTimer();
  abortLiveDemoRequest(true);
  setLiveDemoBusy(false);
  if (elements.messages) elements.messages.innerHTML = '';
  if (elements.promptInput) elements.promptInput.value = '';
  ensureLiveDemoEmptyState();
  setLiveDemoStatus('Context cleared. Enter a new probe.', 'info');
  if (elements.promptInput) elements.promptInput.focus();
}

function triggerLiveDemoPreset(prompt) {
  var elements = getLiveDemoElements();
  if (!elements.promptInput || liveDemoState.isLoading) return;
  clearLiveDemoPresetTimer();
  elements.promptInput.value = prompt;
  elements.promptInput.focus();
  setLiveDemoStatus('Preset loaded. Sending...', 'info');
  liveDemoState.presetTimer = setTrackedTimeout(function() {
    liveDemoState.presetTimer = null;
    submitLiveDemoPrompt();
  }, 120);
}

function initLiveFailureDemo() {
  ensureLiveDemoEmptyState();
  if (!liveDemoState.isLoading) {
    setLiveDemoBusy(false);
    setLiveDemoStatus('Enter an API key, then send a probe.', 'info');
  }
}

function resetLiveFailureDemo() {
  clearLiveDemoPresetTimer();
  abortLiveDemoRequest(true);
  setLiveDemoBusy(false);

  var elements = getLiveDemoElements();
  if (elements.messages) elements.messages.innerHTML = '';
  if (elements.promptInput) elements.promptInput.value = '';

  ensureLiveDemoEmptyState();
  setLiveDemoStatus('Enter an API key, then send a probe.', 'info');
}

async function submitLiveDemoPrompt(explicitPrompt) {
  var elements = getLiveDemoElements();
  if (!elements.root || liveDemoState.isLoading) return;
  clearLiveDemoPresetTimer();

  var apiKey = elements.apiKeyInput && elements.apiKeyInput.value ? elements.apiKeyInput.value.trim() : '';
  var prompt = typeof explicitPrompt === 'string' ? explicitPrompt : (elements.promptInput ? elements.promptInput.value : '');
  prompt = prompt.trim();

  if (!apiKey) {
    setLiveDemoStatus('Enter an API key before sending the request.', 'warn');
    if (elements.apiKeyInput) elements.apiKeyInput.focus();
    return;
  }

  if (!prompt) {
    setLiveDemoStatus('Type a prompt or click one of the preset failure probes.', 'warn');
    if (elements.promptInput) elements.promptInput.focus();
    return;
  }

  var model = LIVE_DEMO_MODEL;
  var conversation = getLiveDemoConversationMessages();

  var userBubble = appendLiveDemoMessage('user', prompt);
  var loadingBubble = appendLiveDemoMessage('assistant', 'Thinking...', { loading: true });
  if (elements.promptInput) elements.promptInput.value = '';

  setLiveDemoBusy(true);
  setLiveDemoStatus('Waiting for ' + model + '...', 'info');

  var controller = new AbortController();
  liveDemoState.abortController = controller;
  var endpoint = LIVE_DEMO_ENDPOINT;

  try {
    var response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + apiKey
      },
      body: JSON.stringify({
        model: model,
        messages: conversation.concat([{ role: 'user', content: prompt }]),
        temperature: 0
      }),
      signal: controller.signal
    });

    var payload = await response.json().catch(function() {
      return null;
    });

    if (!response.ok) {
      var errorMessage = payload && payload.error && payload.error.message
        ? payload.error.message
        : 'Request failed with status ' + response.status + '.';
      throw new Error(errorMessage);
    }

    var reply = getLiveDemoResponseText(payload);
    if (!reply) {
      throw new Error('The model returned an empty reply.');
    }

    if (loadingBubble) {
      loadingBubble.classList.remove('loading');
      loadingBubble.dataset.role = 'assistant';
      var body = loadingBubble.querySelector('.live-demo-message-body');
      if (body) body.textContent = reply;
    }

    setLiveDemoStatus('Response received from ' + model + '.', 'success');
  } catch (error) {
    if (loadingBubble && loadingBubble.parentNode) loadingBubble.parentNode.removeChild(loadingBubble);
    if (userBubble && userBubble.parentNode) userBubble.parentNode.removeChild(userBubble);
    ensureLiveDemoEmptyState();

    if (elements.promptInput) elements.promptInput.value = prompt;

    if (error && error.name === 'AbortError') {
      if (!liveDemoState.suppressAbortStatus) {
        setLiveDemoStatus('Request stopped.', 'warn');
      }
    } else {
      setLiveDemoStatus(error && error.message ? error.message : 'Request failed.', 'warn');
    }
  } finally {
    liveDemoState.abortController = null;
    liveDemoState.suppressAbortStatus = false;
    setLiveDemoBusy(false);
    syncLiveDemoScroll();
    scheduleActiveSlideFit({ reason: 'live-demo-complete', dispatchResize: false });
  }
}
