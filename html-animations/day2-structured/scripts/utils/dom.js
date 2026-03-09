function addTrackedListener(target, type, handler, options) {
  target.addEventListener(type, handler, options);
  if (!DEV) return;
  const targetLabel = target === window
    ? 'window'
    : target === document
      ? 'document'
      : (target.id ? '#' + target.id : (target.tagName || 'unknown'));
  const key = targetLabel + '::' + type;
  const singleRegistrationType = type === 'keydown' || type === 'transitionend';
  if (singleRegistrationType && state.dev.listenerKeys.has(key)) {
    console.warn('[DEV][listener duplicate]', key);
  }
  if (singleRegistrationType) state.dev.listenerKeys.add(key);
  state.dev.listenerCounts[type] = (state.dev.listenerCounts[type] || 0) + 1;
}

function setAttrs(el, attrs) {
  if (!attrs) return el;
  Object.keys(attrs).forEach((key) => {
    const value = attrs[key];
    if (value === null || typeof value === 'undefined') return;
    if (key === 'className') {
      el.className = value;
      return;
    }
    if (key === 'text') {
      el.textContent = value;
      return;
    }
    if (key === 'html') {
      el.innerHTML = value;
      return;
    }
    if (key === 'dataset' && value && typeof value === 'object') {
      Object.keys(value).forEach((name) => {
        el.dataset[name] = value[name];
      });
      return;
    }
    if (key === 'style' && value && typeof value === 'object') {
      Object.keys(value).forEach((name) => {
        el.style[name] = value[name];
      });
      return;
    }
    if (key in el) {
      el[key] = value;
      return;
    }
    el.setAttribute(key, value);
  });
  return el;
}

function appendChildren(el, children) {
  if (children === null || typeof children === 'undefined') return el;
  const nodes = Array.isArray(children) ? children : [children];
  nodes.forEach((child) => {
    if (child === null || typeof child === 'undefined') return;
    if (typeof child === 'string' || typeof child === 'number') {
      el.appendChild(document.createTextNode(String(child)));
      return;
    }
    el.appendChild(child);
  });
  return el;
}

function createEl(tag, attrs, children) {
  const el = document.createElement(tag);
  setAttrs(el, attrs);
  appendChildren(el, children);
  return el;
}

function svgEl(tag, attrs, children) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  setAttrs(el, attrs);
  appendChildren(el, children);
  return el;
}

function createVectorRect(config) {
  const vector = createEl('div', {
    className: config.baseClass,
    id: config.id
  });
  for (let i = 1; i <= 3; i += 1) {
    vector.appendChild(createEl('span', {
      className: config.dividerClass,
      style: {
        left: (i * 25) + '%'
      }
    }));
  }
  return vector;
}

function getPathLength(path) {
  if (!path || typeof path.getTotalLength !== 'function') return 0;
  try {
    return path.getTotalLength();
  } catch (err) {
    return 0;
  }
}

function hideArrowElements(config) {
  const path = config.path;
  if (!path) return;
  const length = getPathLength(path);
  const dash = length > 0 ? length.toFixed(2) : '0';
  path.style.transition = 'none';
  path.style.strokeDasharray = dash;
  path.style.strokeDashoffset = dash;
  if (config.head) {
    config.head.style.transition = 'none';
    config.head.style.opacity = '0';
  }
  if (config.dot) {
    config.dot.style.transition = 'none';
    config.dot.style.opacity = '0';
  }
}

function showArrowElements(config) {
  const path = config.path;
  if (!path) return;
  const length = getPathLength(path);
  const dash = length > 0 ? length.toFixed(2) : '0';
  path.style.transition = 'none';
  path.style.strokeDasharray = dash;
  path.style.strokeDashoffset = '0';
  if (config.head) {
    config.head.style.transition = 'none';
    config.head.style.opacity = '1';
  }
  if (config.dot) {
    config.dot.style.transition = 'none';
    config.dot.style.opacity = '1';
  }
}

function cacheUiReferences() {
  state.ui.btnPrev = $('#btnPrev');
  state.ui.btnNext = $('#btnNext');
  state.ui.btnSkip = $('#btnSkip');
  state.ui.slideCounter = $('#slideCounter');
  state.ui.progressFill = $('#progressFill');
}

