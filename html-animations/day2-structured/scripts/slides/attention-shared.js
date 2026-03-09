function formatMathToken(token) {
  return '\\mathrm{' + String(token) + '}';
}

function formatMathSubscript(base, sub) {
  return base + '_{' + formatMathToken(sub) + '}';
}

function formatTokenMath(base, token) {
  return inlineMath(formatMathSubscript(base, token));
}

function createAttentionQkvVectorRect(vectorId, extraClass = '', values = null) {
  return createVectorRect({
    id: vectorId,
    baseClass: 'attn19-vector' + (extraClass ? ' ' + extraClass : ''),
    dividerClass: 'attn19-vector-divider',
    values
  });
}

function createMathSubLabel(base, sub, className) {
  return createEl('span', {
    className,
    html: formatTokenMath(base, sub)
  });
}

function createMathQTransposeK() {
  return createEl('span', {
    html: inlineMath('q^{\\mathsf{T}}k')
  });
}

function createAttentionStep4VectorRect(vectorId, extraClass = '', values = null) {
  return createVectorRect({
    id: vectorId,
    baseClass: 'attn19-vector' + (extraClass ? ' ' + extraClass : ''),
    dividerClass: 'attn19-vector-divider',
    values
  });
}

function createAttentionSkeletonVectorRect(vectorId, extraClass = '', dims = 4) {
  const vector = createEl('div', {
    className: 'attn19-vector' + (extraClass ? ' ' + extraClass : ''),
    id: vectorId
  });
  for (let i = 1; i < dims; i += 1) {
    vector.appendChild(createEl('span', {
      className: 'attn19-vector-divider',
      style: {
        left: ((i / dims) * 100) + '%'
      }
    }));
  }
  return vector;
}
