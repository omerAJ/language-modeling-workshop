function createAttentionMatrixSourceItem(token) {
  const item = createEl('div', {
    className: 'attn23-source-item',
    id: 'attn23-source-item-' + token,
    dataset: { token }
  });

  const chipWrap = createEl('div', {
    className: 'attn19-chip-wrap attn23-source-chip-wrap',
    id: 'attn23-source-chip-wrap-' + token
  });
  chipWrap.appendChild(createEl('div', {
    className: 'attn19-token-chip',
    id: 'attn23-source-chip-' + token,
    text: token
  }));
  item.appendChild(chipWrap);

  const xWrap = createEl('div', {
    className: 'attn19-vector-wrap attn19-vector-wrap-x attn23-source-vector-wrap',
    id: 'attn23-source-vector-wrap-' + token
  }, [
    createMathSubLabel('x', token, 'attn19-vector-label'),
    createAttentionQkvVectorRect(
      'attn23-source-vector-' + token,
      '',
      ATTN_QKV_X_VECTORS[token] || ATTN_QKV_QUERY_VECTOR
    )
  ]);
  item.appendChild(xWrap);

  return item;
}

function createAttentionMatrixTokenItem(token) {
  const item = createEl('div', {
    className: 'attn23-token-item',
    id: 'attn23-token-item-' + token,
    dataset: { token }
  });

  const chipWrap = createEl('div', {
    className: 'attn19-chip-wrap attn23-token-chip-wrap',
    id: 'attn23-token-chip-wrap-' + token
  });
  chipWrap.appendChild(createEl('div', {
    className: 'attn19-token-chip',
    id: 'attn23-token-chip-' + token,
    text: token
  }));
  item.appendChild(chipWrap);

  return item;
}

function createAttentionMatrixXItem(token) {
  const item = createEl('div', {
    className: 'attn23-x-item',
    id: 'attn23-x-item-' + token,
    dataset: { token }
  });

  const xWrap = createEl('div', {
    className: 'attn19-vector-wrap attn19-vector-wrap-x attn23-x-vector-wrap',
    id: 'attn23-x-vector-wrap-' + token
  }, createAttentionQkvVectorRect(
    'attn23-x-vector-' + token,
    '',
    ATTN_QKV_X_VECTORS[token] || ATTN_QKV_QUERY_VECTOR
  ));
  item.appendChild(xWrap);

  return item;
}

function createAttentionMatrixMiniX(proj) {
  const wrap = createEl('div', {
    className: 'attn23-mini-x-wrap',
    id: 'attn23-mini-x-wrap-' + proj
  });
  wrap.appendChild(createEl('div', {
    className: 'attn23-mini-x-label',
    html: inlineMath('X')
  }));
  const shell = createEl('div', {
    className: 'attn23-mini-x-shell',
    id: 'attn23-mini-x-shell-' + proj
  });
  const slots = createEl('div', {
    className: 'attn23-mini-x-slots',
    id: 'attn23-mini-x-slots-' + proj
  });
  ATTN_MATRIX_TOKENS.forEach((token) => {
    const row = createEl('div', { className: 'attn23-mini-x-row' });
    row.appendChild(createEl('div', {
      className: 'attn19-vector-wrap attn19-vector-wrap-x'
    }, createAttentionQkvVectorRect(
      'attn23-mini-x-vector-' + proj + '-' + token,
      ''
    )));
    slots.appendChild(row);
  });
  shell.appendChild(slots);
  wrap.appendChild(shell);
  return wrap;
}

function createAttentionMatrixProjectionMatrix(proj) {
  const cluster = createEl('div', {
    className: 'attn23-proj-matrix-cluster',
    id: 'attn23-proj-matrix-cluster-' + proj
  });
  const shell = createEl('div', {
    className: 'attn23-proj-matrix-shell',
    id: 'attn23-proj-matrix-shell-' + proj
  });
  const matrix = createEl('div', {
    className: 'attn19p1-matrix',
    id: 'attn23-proj-matrix-' + proj
  });
  for (let i = 1; i <= 3; i += 1) {
    matrix.appendChild(createEl('span', {
      className: 'attn19p1-matrix-grid-v',
      style: { left: (i * 25) + '%' }
    }));
    matrix.appendChild(createEl('span', {
      className: 'attn19p1-matrix-grid-h',
      style: { top: (i * 25) + '%' }
    }));
  }
  shell.appendChild(matrix);
  cluster.appendChild(shell);
  cluster.appendChild(createEl('div', {
    className: 'attn19p1-matrix-label',
    id: 'attn23-proj-matrix-label-' + proj,
    html: inlineMath('W_' + proj.toUpperCase())
  }));
  return cluster;
}

function createAttentionMatrixProjectionOutputRow(proj, token) {
  const row = createEl('div', {
    className: 'attn23-proj-output-row',
    dataset: { proj, token }
  });
  row.appendChild(createEl('div', {
    className: 'attn19-vector-wrap attn19-vector-wrap-x'
  }, createAttentionQkvVectorRect(
    'attn23-proj-output-vector-' + proj + '-' + token,
    ''
  )));
  return row;
}

function createAttentionMatrixProjectionOutput(proj) {
  const wrap = createEl('div', {
    className: 'attn23-proj-output-wrap',
    id: 'attn23-proj-output-wrap-' + proj
  });
  wrap.appendChild(createEl('div', {
    className: 'attn23-proj-output-label',
    id: 'attn23-proj-output-label-' + proj,
    dataset: { proj },
    html: inlineMath(ATTN_MATRIX_OUTPUT_LABELS[proj])
  }));
  const shell = createEl('div', {
    className: 'attn23-proj-output-shell',
    id: 'attn23-proj-output-shell-' + proj
  });
  const slots = createEl('div', {
    className: 'attn23-proj-output-slots',
    id: 'attn23-proj-output-slots-' + proj
  });
  ATTN_MATRIX_TOKENS.forEach((token) => {
    slots.appendChild(createAttentionMatrixProjectionOutputRow(proj, token));
  });
  shell.appendChild(slots);
  wrap.appendChild(shell);
  return wrap;
}

function createAttentionMatrixProjectionColumn(proj) {
  const col = createEl('div', {
    className: 'attn23-proj-col',
    id: 'attn23-proj-col-' + proj,
    dataset: { proj }
  });

  const mulRow = createEl('div', {
    className: 'attn23-proj-mul-row',
    id: 'attn23-proj-mul-row-' + proj
  });
  mulRow.appendChild(createAttentionMatrixMiniX(proj));
  mulRow.appendChild(createEl('span', {
    className: 'attn19p1-mul-symbol attn23-proj-mul-symbol',
    text: '\u00d7'
  }));
  mulRow.appendChild(createAttentionMatrixProjectionMatrix(proj));

  col.appendChild(mulRow);
  col.appendChild(createAttentionMatrixProjectionOutput(proj));
  return col;
}

function createAttentionMatrixScoreStructRow(prefix, token, vectorClass = '') {
  const row = createEl('div', {
    className: 'attn23-score-struct-row',
    id: prefix + '-row-' + token,
    dataset: { token }
  });
  row.appendChild(createEl('div', {
    className: 'attn19-vector-wrap attn19-vector-wrap-x'
  }, createAttentionQkvVectorRect(
    prefix + '-vector-' + token,
    vectorClass,
    null
  )));
  return row;
}

function createAttentionMatrixScoreQMatrix() {
  const wrap = createEl('div', {
    className: 'attn23-score-matrix-wrap attn23-score-q-wrap',
    id: 'attn23-score-q-wrap'
  });
  wrap.appendChild(createEl('div', {
    className: 'attn23-score-matrix-label',
    id: 'attn23-score-q-label',
    html: inlineMath('Q')
  }));
  wrap.appendChild(createEl('div', {
    className: 'attn23-score-dims',
    id: 'attn23-score-q-dims',
    html: inlineMath('Q \\in \\mathbb{R}^{S \\times d_k}')
  }));

  const layout = createEl('div', {
    className: 'attn23-score-struct-layout attn23-score-q-layout'
  });
  const rowHeaders = createEl('div', {
    className: 'attn23-score-row-headers attn23-score-q-row-headers'
  });
  const shell = createEl('div', {
    className: 'attn23-score-shell attn23-score-struct-shell attn23-score-q-shell',
    id: 'attn23-score-q-shell'
  });
  const slots = createEl('div', {
    className: 'attn23-score-struct-slots',
    id: 'attn23-score-q-body'
  });

  ATTN_MATRIX_SCORE_TOKENS.forEach((token) => {
    rowHeaders.appendChild(createEl('span', {
      className: 'attn23-score-token-header',
      text: token
    }));
    slots.appendChild(createAttentionMatrixScoreStructRow('attn23-score-q', token, 'attn23-score-struct-vector-q'));
  });

  shell.appendChild(slots);
  layout.appendChild(rowHeaders);
  layout.appendChild(shell);
  wrap.appendChild(layout);
  return wrap;
}

function createAttentionMatrixScoreKTransposeMatrix() {
  const wrap = createEl('div', {
    className: 'attn23-score-matrix-wrap attn23-score-k-wrap',
    id: 'attn23-score-k-wrap'
  });
  wrap.appendChild(createEl('div', {
    className: 'attn23-score-matrix-label',
    id: 'attn23-score-k-label',
    html: inlineMath('K')
  }));
  wrap.appendChild(createEl('div', {
    className: 'attn23-score-dims',
    id: 'attn23-score-k-dims',
    html: inlineMath('K \\in \\mathbb{R}^{S \\times d_k}')
  }));

  const stage = createEl('div', {
    className: 'attn23-score-k-stage',
    id: 'attn23-score-k-stage'
  });
  const colHeaders = createEl('div', {
    className: 'attn23-score-col-headers attn23-score-k-col-headers',
    id: 'attn23-score-k-col-headers'
  });
  ATTN_MATRIX_SCORE_TOKENS.forEach((token) => {
    colHeaders.appendChild(createEl('span', {
      className: 'attn23-score-token-header',
      text: token
    }));
  });

  const bodyArea = createEl('div', {
    className: 'attn23-score-k-body-area',
    id: 'attn23-score-k-body-area'
  });
  const rowHeaders = createEl('div', {
    className: 'attn23-score-row-headers attn23-score-k-row-headers',
    id: 'attn23-score-k-row-headers'
  });
  ATTN_MATRIX_SCORE_TOKENS.forEach((token) => {
    rowHeaders.appendChild(createEl('span', {
      className: 'attn23-score-token-header',
      text: token
    }));
  });

  const frame = createEl('div', {
    className: 'attn23-score-k-body-frame',
    id: 'attn23-score-k-body-frame'
  });
  const rotate = createEl('div', {
    className: 'attn23-score-k-rotate',
    id: 'attn23-score-k-rotate'
  });
  const shell = createEl('div', {
    className: 'attn23-score-shell attn23-score-struct-shell attn23-score-k-shell',
    id: 'attn23-score-k-shell'
  });
  const slots = createEl('div', {
    className: 'attn23-score-struct-slots',
    id: 'attn23-score-k-body'
  });

  ATTN_MATRIX_SCORE_TOKENS.forEach((token) => {
    slots.appendChild(createAttentionMatrixScoreStructRow('attn23-score-k', token, 'attn23-score-struct-vector-k'));
  });

  shell.appendChild(slots);
  rotate.appendChild(shell);
  frame.appendChild(rotate);
  bodyArea.appendChild(rowHeaders);
  bodyArea.appendChild(frame);
  stage.appendChild(colHeaders);
  stage.appendChild(bodyArea);
  wrap.appendChild(stage);
  return wrap;
}

function createAttentionMatrixScoreCell(rowToken, colToken, value, mode = 'score') {
  const rowIdx = ATTN_MATRIX_SCORE_TOKENS.indexOf(rowToken);
  const colIdx = ATTN_MATRIX_SCORE_TOKENS.indexOf(colToken);
  const cell = createEl('div', {
    className: 'attn23-score-cell',
    id: 'attn23-score-cell-' + rowToken + '-' + colToken,
    dataset: {
      rowToken,
      colToken,
      future: String(colIdx > rowIdx),
      diagonal: String(colIdx === rowIdx)
    }
  });
  cell.appendChild(createEl('span', {
    className: 'attn23-score-cell-value',
    text: formatAttentionMatrixScoreCellValue(value, mode)
  }));
  cell.appendChild(createEl('span', {
    className: 'attn23-score-cell-mask',
    'aria-hidden': 'true'
  }));
  return cell;
}

function createAttentionMatrixScoreMatrix() {
  const wrap = createEl('div', {
    className: 'attn23-score-matrix-wrap attn23-score-s-wrap',
    id: 'attn23-score-s-wrap'
  });
  wrap.appendChild(createEl('div', {
    className: 'attn23-score-matrix-label',
    id: 'attn23-score-s-label',
    html: 'Raw Scores \\(S\\)'
  }));
  wrap.appendChild(createEl('div', {
    className: 'attn23-score-dims',
    id: 'attn23-score-s-dims',
    html: inlineMath('S \\in \\mathbb{R}^{S \\times S}')
  }));

  const layout = createEl('div', {
    className: 'attn23-score-s-layout'
  });
  layout.appendChild(createEl('div', {
    className: 'attn23-score-s-corner'
  }));

  const colHeaders = createEl('div', {
    className: 'attn23-score-col-headers attn23-score-s-col-headers'
  });
  ATTN_MATRIX_SCORE_TOKENS.forEach((token) => {
    colHeaders.appendChild(createEl('span', {
      className: 'attn23-score-token-header',
      text: token
    }));
  });
  layout.appendChild(colHeaders);

  const rowHeaders = createEl('div', {
    className: 'attn23-score-row-headers attn23-score-s-row-headers'
  });
  const shell = createEl('div', {
    className: 'attn23-score-shell attn23-score-grid-shell',
    id: 'attn23-score-matrix-shell'
  });
  const grid = createEl('div', {
    className: 'attn23-score-grid',
    id: 'attn23-score-matrix-grid'
  });

  ATTN_MATRIX_SCORE_TOKENS.forEach((rowToken) => {
    rowHeaders.appendChild(createEl('span', {
      className: 'attn23-score-token-header',
      text: rowToken
    }));

    const row = createEl('div', {
      className: 'attn23-score-row',
      id: 'attn23-score-row-' + rowToken,
      dataset: { rowToken }
    });
    ATTN_MATRIX_SCORE_TOKENS.forEach((colToken) => {
      row.appendChild(createAttentionMatrixScoreCell(
        rowToken,
        colToken,
        (ATTN_MATRIX_SCORE_RAW_ROWS[rowToken] || {})[colToken],
        'score'
      ));
    });
    grid.appendChild(row);
  });

  shell.appendChild(grid);
  layout.appendChild(rowHeaders);
  layout.appendChild(shell);
  wrap.appendChild(layout);
  return wrap;
}

function createAttentionMatrixScoreStage() {
  const workspace = createEl('div', {
    className: 'attn23-score-workspace',
    id: 'attn23-score-workspace'
  });
  const formulaRow = createEl('div', {
    className: 'attn23-score-formula-row',
    id: 'attn23-score-formula-row'
  });

  formulaRow.appendChild(createAttentionMatrixScoreQMatrix());
  formulaRow.appendChild(createEl('div', {
    className: 'attn23-score-mul',
    id: 'attn23-score-mul',
    text: '\u00d7'
  }));
  formulaRow.appendChild(createAttentionMatrixScoreKTransposeMatrix());
  formulaRow.appendChild(createEl('div', {
    className: 'attn23-score-equals',
    id: 'attn23-score-equals',
    text: '='
  }));
  formulaRow.appendChild(createAttentionMatrixScoreMatrix());

  workspace.appendChild(formulaRow);
  workspace.appendChild(createAttentionMatrixPostScoreStage());
  return workspace;
}

function createAttentionMatrixPostScoreCell(rowToken, colToken, value, mode = 'score') {
  const rowIdx = ATTN_MATRIX_SCORE_TOKENS.indexOf(rowToken);
  const colIdx = ATTN_MATRIX_SCORE_TOKENS.indexOf(colToken);
  const cell = createEl('div', {
    className: 'attn23-score-cell attn23-postscore-cell',
    id: 'attn23-postscore-cell-' + rowToken + '-' + colToken,
    dataset: {
      rowToken,
      colToken,
      future: String(colIdx > rowIdx),
      diagonal: String(colIdx === rowIdx)
    }
  });
  cell.appendChild(createEl('span', {
    className: 'attn23-score-cell-value',
    text: formatAttentionMatrixScoreCellValue(value, mode)
  }));
  cell.appendChild(createEl('span', {
    className: 'attn23-score-cell-mask',
    'aria-hidden': 'true'
  }));
  return cell;
}

function createAttentionMatrixPostScoreMainMatrix() {
  const wrap = createEl('div', {
    className: 'attn23-postscore-main-wrap',
    id: 'attn23-postscore-main-wrap'
  });
  wrap.appendChild(createEl('div', {
    className: 'attn23-score-matrix-label attn23-postscore-main-label',
    id: 'attn23-postscore-main-label',
    html: 'Masked Scores \\(S\\)'
  }));
  wrap.appendChild(createEl('div', {
    className: 'attn23-score-dims attn23-postscore-main-dims',
    id: 'attn23-postscore-main-dims',
    html: inlineMath('S \\in \\mathbb{R}^{S \\times S}')
  }));

  const layout = createEl('div', {
    className: 'attn23-score-s-layout'
  });
  layout.appendChild(createEl('div', {
    className: 'attn23-score-s-corner'
  }));

  const colHeaders = createEl('div', {
    className: 'attn23-score-col-headers attn23-score-s-col-headers'
  });
  ATTN_MATRIX_SCORE_TOKENS.forEach((token) => {
    colHeaders.appendChild(createEl('span', {
      className: 'attn23-score-token-header',
      text: token
    }));
  });
  layout.appendChild(colHeaders);

  const rowHeaders = createEl('div', {
    className: 'attn23-score-row-headers attn23-score-s-row-headers'
  });
  const shell = createEl('div', {
    className: 'attn23-score-shell attn23-score-grid-shell attn23-postscore-main-shell',
    id: 'attn23-postscore-main-shell'
  });
  const grid = createEl('div', {
    className: 'attn23-score-grid attn23-postscore-main-grid',
    id: 'attn23-postscore-main-grid'
  });

  ATTN_MATRIX_SCORE_TOKENS.forEach((rowToken) => {
    rowHeaders.appendChild(createEl('span', {
      className: 'attn23-score-token-header',
      text: rowToken
    }));
    const row = createEl('div', {
      className: 'attn23-postscore-row',
      id: 'attn23-postscore-row-' + rowToken,
      dataset: { rowToken }
    });
    ATTN_MATRIX_SCORE_TOKENS.forEach((colToken) => {
      row.appendChild(createAttentionMatrixPostScoreCell(
        rowToken,
        colToken,
        (ATTN_MATRIX_SCORE_RAW_MASKED_ROWS[rowToken] || {})[colToken],
        ((ATTN_MATRIX_CAUSAL_MASK[rowToken] || {})[colToken]) ? 'masked' : 'score'
      ));
    });
    grid.appendChild(row);
  });

  shell.appendChild(grid);
  layout.appendChild(rowHeaders);
  layout.appendChild(shell);
  wrap.appendChild(layout);
  return wrap;
}

function createAttentionMatrixPostScoreVectorRow(prefix, token, values, rowClass = '') {
  return createEl('div', {
    className: 'attn23-proj-output-row attn23-postscore-vector-row ' + rowClass,
    id: prefix + '-row-' + token,
    dataset: { token }
  }, createEl('div', {
    className: 'attn19-vector-wrap attn19-vector-wrap-x'
  }, createAttentionQkvVectorRect(
    prefix + '-vector-' + token,
    '',
    values
  )));
}

function createAttentionMatrixPostScoreValueMatrix() {
  const wrap = createEl('div', {
    className: 'attn23-postscore-v-wrap',
    id: 'attn23-postscore-v-wrap'
  });
  wrap.appendChild(createEl('div', {
    className: 'attn23-score-matrix-label attn23-postscore-v-label',
    id: 'attn23-postscore-v-label',
    html: 'Value Matrix \\(V\\)'
  }));
  wrap.appendChild(createEl('div', {
    className: 'attn23-score-dims',
    id: 'attn23-postscore-v-dims',
    html: inlineMath('V \\in \\mathbb{R}^{S \\times d_v}')
  }));
  const shell = createEl('div', {
    className: 'attn23-postscore-vector-shell',
    id: 'attn23-postscore-v-shell'
  });
  const slots = createEl('div', {
    className: 'attn23-postscore-vector-slots',
    id: 'attn23-postscore-v-slots'
  });
  ATTN_MATRIX_TOKENS.forEach((token) => {
    slots.appendChild(createAttentionMatrixPostScoreVectorRow(
      'attn23-postscore-v',
      token,
      ATTN_QKV_VALUE_VECTORS[token] || [],
      'attn23-postscore-v-row'
    ));
  });
  shell.appendChild(slots);
  wrap.appendChild(shell);
  return wrap;
}

function createAttentionMatrixPostScoreOutputMatrix() {
  const wrap = createEl('div', {
    className: 'attn23-postscore-o-wrap',
    id: 'attn23-postscore-o-wrap'
  });
  wrap.appendChild(createEl('div', {
    className: 'attn23-score-matrix-label attn23-postscore-o-label',
    id: 'attn23-postscore-o-label',
    html: 'Output Matrix \\(O\\)'
  }));
  wrap.appendChild(createEl('div', {
    className: 'attn23-score-dims',
    id: 'attn23-postscore-o-dims',
    html: inlineMath('O \\in \\mathbb{R}^{S \\times d_v}')
  }));
  const shell = createEl('div', {
    className: 'attn23-postscore-vector-shell',
    id: 'attn23-postscore-o-shell'
  });
  const slots = createEl('div', {
    className: 'attn23-postscore-vector-slots',
    id: 'attn23-postscore-o-slots'
  });
  ATTN_MATRIX_TOKENS.forEach((token) => {
    const row = createAttentionMatrixPostScoreVectorRow(
      'attn23-postscore-o',
      token,
      ATTN_MATRIX_OUTPUT_ROWS[token] || [],
      'attn23-postscore-o-row'
    );
    row.classList.add('is-hidden');
    slots.appendChild(row);
  });
  shell.appendChild(slots);
  wrap.appendChild(shell);
  return wrap;
}

function createAttentionMatrixPostScoreStage() {
  const stage = createEl('div', {
    className: 'attn23-postscore-stage',
    id: 'attn23-postscore-stage'
  });
  stage.appendChild(createAttentionMatrixPostScoreMainMatrix());
  stage.appendChild(createEl('div', {
    className: 'attn23-postscore-op attn23-postscore-scale-op',
    id: 'attn23-postscore-scale-op',
    html: '\\(/ \\sqrt{d_k}\\)'
  }));
  stage.appendChild(createEl('div', {
    className: 'attn23-postscore-op attn23-postscore-softmax-op',
    id: 'attn23-postscore-softmax-op'
  }, [
    createEl('span', {
      className: 'attn23-postscore-softmax-main',
      html: inlineMath('\\operatorname{softmax}(z_i)_j = \\frac{\\exp(z_{ij})}{\\sum_k \\exp(z_{ik})}')
    }),
    createEl('span', {
      className: 'attn23-postscore-softmax-note',
      html: '\\(\\exp(-\\infty) = 0\\) and masked cells become \\(0\\).'
    })
  ]));
  stage.appendChild(createEl('div', {
    className: 'attn23-postscore-row-sum',
    id: 'attn23-postscore-row-sum',
    html: inlineMath('\\sum_j a_{ij} = 1')
  }));
  stage.appendChild(createEl('div', {
    className: 'attn23-postscore-mul',
    id: 'attn23-postscore-mul',
    text: '\u00d7'
  }));
  stage.appendChild(createAttentionMatrixPostScoreValueMatrix());
  stage.appendChild(createEl('div', {
    className: 'attn23-postscore-equals',
    id: 'attn23-postscore-equals',
    text: '='
  }));
  stage.appendChild(createAttentionMatrixPostScoreOutputMatrix());
  stage.appendChild(createEl('div', {
    className: 'attn23-postscore-caption',
    id: 'attn23-postscore-caption',
    html: inlineMath('o_i = \\sum_j a_{ij} v_j')
  }));
  return stage;
}

function clearAttentionMatrixTimers() {
  state.attentionMatrix.timers.forEach((timerId) => clearTimeout(timerId));
  state.attentionMatrix.timers = [];
  state.attentionMatrix.rafIds.forEach((rafId) => cancelAnimationFrame(rafId));
  state.attentionMatrix.rafIds = [];
}

function clearAttentionMatrixProjectionTimers() {
  state.attentionMatrix.projectionTimers.forEach((timerId) => clearTimeout(timerId));
  state.attentionMatrix.projectionTimers = [];
  state.attentionMatrix.projectionRafIds.forEach((rafId) => cancelAnimationFrame(rafId));
  state.attentionMatrix.projectionRafIds = [];
}

function clearAttentionMatrixScoreTimers() {
  state.attentionMatrix.scoreTimers.forEach((timerId) => clearTimeout(timerId));
  state.attentionMatrix.scoreTimers = [];
  state.attentionMatrix.scoreRafIds.forEach((rafId) => cancelAnimationFrame(rafId));
  state.attentionMatrix.scoreRafIds = [];
}

function clearAttentionMatrixGhosts() {
  const layer = document.getElementById('attn23-ghost-layer');
  if (!layer) return;
  layer.querySelectorAll('.attn23-ghost').forEach((ghost) => ghost.remove());
}

function setAttentionMatrixTokenSlotVisible(token, visible) {
  const slot = document.getElementById('attn23-token-slot-' + token);
  if (!slot) return;
  slot.classList.toggle('is-visible', !!visible);
}

function setAttentionMatrixXSlotVisible(token, visible) {
  const slot = document.getElementById('attn23-x-slot-' + token);
  if (!slot) return;
  slot.classList.toggle('is-visible', !!visible);
}

function syncAttentionMatrixTokenSlots(visibleCount = state.attentionMatrix.tokenVisibleCount) {
  const clamped = Math.max(0, Math.min(ATTN_MATRIX_TOKENS.length, visibleCount));
  ATTN_MATRIX_TOKENS.forEach((token, idx) => {
    setAttentionMatrixTokenSlotVisible(token, idx < clamped);
  });
}

function syncAttentionMatrixXSlots(visibleCount = state.attentionMatrix.xVisibleCount) {
  const clamped = Math.max(0, Math.min(ATTN_MATRIX_TOKENS.length, visibleCount));
  ATTN_MATRIX_TOKENS.forEach((token, idx) => {
    setAttentionMatrixXSlotVisible(token, idx < clamped);
  });
}

function syncAttentionMatrixScoreRows(visibleCount = state.attentionMatrix.scoreVisibleCount) {
  const clamped = Math.max(0, Math.min(ATTN_MATRIX_SCORE_TOKENS.length, visibleCount));
  ATTN_MATRIX_SCORE_TOKENS.forEach((token, idx) => {
    const row = document.getElementById('attn23-score-row-' + token);
    if (!row) return;
    row.classList.toggle('is-visible', idx < clamped);
  });
}

function setAttentionMatrixScoreMaskRowState(token, mode = 'none') {
  const row = document.getElementById('attn23-score-row-' + token);
  if (!row) return;
  row.classList.remove('is-mask-problem-visible', 'is-mask-applied-visible', 'is-mask-question-visible');
  if (mode === 'problem') row.classList.add('is-mask-problem-visible');
  if (mode === 'mask') row.classList.add('is-mask-applied-visible');
  if (mode === 'question') row.classList.add('is-mask-question-visible');
}

function syncAttentionMatrixScoreMaskRows(mode = state.attentionMatrix.maskMode, visibleCount = state.attentionMatrix.maskVisibleCount) {
  const clamped = Math.max(0, Math.min(ATTN_MATRIX_SCORE_TOKENS.length, visibleCount));
  ATTN_MATRIX_SCORE_TOKENS.forEach((token, idx) => {
    setAttentionMatrixScoreMaskRowState(token, idx < clamped ? mode : 'none');
  });
  syncAttentionMatrixScoreMaskCellStyles();
  syncAttentionMatrixScoreCellText(mode);
}

function syncAttentionMatrixScoreCellText(mode = state.attentionMatrix.maskMode) {
  const slide = document.getElementById('slide-23');
  if (!slide) return;
  slide.querySelectorAll('.attn23-score-cell').forEach((cell) => {
    const value = cell.querySelector('.attn23-score-cell-value');
    if (!value) return;
    const row = cell.closest('.attn23-score-row');
    const rowToken = cell.dataset.rowToken;
    const colToken = cell.dataset.colToken;
    const future = cell.dataset.future === 'true';
    const rawValue = ((ATTN_MATRIX_SCORE_RAW_ROWS[rowToken] || {})[colToken]);
    const rowMasked = !!(row && (
      row.classList.contains('is-mask-applied-visible')
      || row.classList.contains('is-mask-question-visible')
    ));
    const showMaskedValue = (mode === 'mask' || mode === 'question') && future && rowMasked;
    const displayValue = showMaskedValue
      ? Number.NEGATIVE_INFINITY
      : rawValue;
    const displayMode = showMaskedValue ? 'masked' : 'score';
    value.textContent = formatAttentionMatrixScoreCellValue(displayValue, displayMode);
    value.classList.remove('is-dimmed');
  });
}

function syncAttentionMatrixPostScoreRows(visibleCount = state.attentionMatrix.postScoreVisibleCount) {
  const clamped = Math.max(0, Math.min(ATTN_MATRIX_SCORE_TOKENS.length, visibleCount));
  ATTN_MATRIX_SCORE_TOKENS.forEach((token, idx) => {
    const row = document.getElementById('attn23-postscore-row-' + token);
    if (!row) return;
    row.classList.toggle('is-softmax-done', state.attentionMatrix.postScoreMode === 'attention' && idx < clamped);
  });
}

function syncAttentionMatrixAttentionCellFills() {
  const processed = Math.max(0, Math.min(ATTN_MATRIX_SCORE_TOKENS.length, state.attentionMatrix.postScoreVisibleCount));
  ATTN_MATRIX_SCORE_TOKENS.forEach((rowToken, rowIdx) => {
    ATTN_MATRIX_SCORE_TOKENS.forEach((colToken) => {
      const cell = document.getElementById('attn23-postscore-cell-' + rowToken + '-' + colToken);
      if (!cell) return;
      const future = cell.dataset.future === 'true';
      const isAttentionRow = state.attentionMatrix.postScoreMode === 'attention' && rowIdx < processed;
      const weight = ((ATTN_MATRIX_ATTN_ROWS[rowToken] || {})[colToken]) || 0;
      cell.classList.toggle('is-attention-cell', isAttentionRow);
      cell.style.setProperty('--attn23-postscore-fill-opacity', isAttentionRow && !future ? String(Math.max(0, Math.min(1, weight))) : '0');
      if (future && isAttentionRow) {
        cell.classList.add('is-attention-cell');
      }
      if (!isAttentionRow) {
        cell.classList.remove('is-attention-cell');
      }
    });
  });
}

function syncAttentionMatrixPostScoreMatrix(mode = state.attentionMatrix.postScoreMode) {
  const label = document.getElementById('attn23-postscore-main-label');
  const dims = document.getElementById('attn23-postscore-main-dims');
  const processed = Math.max(0, Math.min(ATTN_MATRIX_SCORE_TOKENS.length, state.attentionMatrix.postScoreVisibleCount));
  if (label) {
    setMathHTML(label, mode === 'attention'
      ? 'Attention Matrix \\(A\\)'
      : (mode === 'scaled' ? 'Scaled Scores \\(Z\\)' : 'Masked Scores \\(S\\)'));
  }
  if (dims) {
    setMathHTML(dims, inlineMath(mode === 'attention'
      ? 'A \\in \\mathbb{R}^{S \\times S}'
      : ((mode === 'scaled') ? 'Z \\in \\mathbb{R}^{S \\times S}' : 'S \\in \\mathbb{R}^{S \\times S}')));
  }

  ATTN_MATRIX_SCORE_TOKENS.forEach((rowToken, rowIdx) => {
    ATTN_MATRIX_SCORE_TOKENS.forEach((colToken) => {
      const cell = document.getElementById('attn23-postscore-cell-' + rowToken + '-' + colToken);
      if (!cell) return;
      const valueEl = cell.querySelector('.attn23-score-cell-value');
      const maskEl = cell.querySelector('.attn23-score-cell-mask');
      const future = cell.dataset.future === 'true';
      if (!valueEl) return;

      let displayValue = ((ATTN_MATRIX_SCORE_RAW_MASKED_ROWS[rowToken] || {})[colToken]);
      let displayMode = future ? 'masked' : 'score';
      let showMask = future;

      if (mode === 'scaled') {
        displayValue = ((ATTN_MATRIX_SCORE_MASKED_ROWS[rowToken] || {})[colToken]);
        displayMode = future ? 'masked' : 'score';
        showMask = future;
      } else if (mode === 'attention') {
        const useAttention = rowIdx < processed;
        if (useAttention) {
          displayValue = ((ATTN_MATRIX_ATTN_ROWS[rowToken] || {})[colToken]) || 0;
          displayMode = 'probability';
          showMask = false;
        } else {
          displayValue = ((ATTN_MATRIX_SCORE_MASKED_ROWS[rowToken] || {})[colToken]);
          displayMode = future ? 'masked' : 'score';
          showMask = future;
        }
      }

      valueEl.textContent = formatAttentionMatrixScoreCellValue(displayValue, displayMode);
      cell.classList.toggle('is-masked', showMask);
      if (maskEl) maskEl.classList.toggle('is-visible', showMask);
    });
  });

  syncAttentionMatrixAttentionCellFills();
}

function syncAttentionMatrixOutputRows(visibleCount = state.attentionMatrix.outputVisibleCount) {
  const clamped = Math.max(0, Math.min(ATTN_MATRIX_TOKENS.length, visibleCount));
  ATTN_MATRIX_TOKENS.forEach((token, idx) => {
    const row = document.getElementById('attn23-postscore-o-row-' + token);
    if (!row) return;
    row.classList.toggle('is-hidden', idx >= clamped);
  });
}

function resetAttentionMatrixPostScoreVisuals() {
  const slide = document.getElementById('slide-23');
  state.attentionMatrix.postScoreCenteredDone = false;
  state.attentionMatrix.scaledMatrixDone = false;
  state.attentionMatrix.attentionMatrixDone = false;
  state.attentionMatrix.valueMatrixVisibleDone = false;
  state.attentionMatrix.outputMatrixDone = false;
  state.attentionMatrix.postScoreMode = 'none';
  state.attentionMatrix.postScoreVisibleCount = 0;
  state.attentionMatrix.outputVisibleCount = 0;
  clearAttentionMatrixPostScoreRowActivity();
  ATTN_MATRIX_SCORE_TOKENS.forEach((token) => {
    const row = document.getElementById('attn23-postscore-row-' + token);
    if (!row) return;
    row.classList.remove('is-softmax-done');
  });
  const vWrap = document.getElementById('attn23-postscore-v-wrap');
  if (vWrap) vWrap.classList.remove('is-source-emphasis');
  const rowSum = document.getElementById('attn23-postscore-row-sum');
  if (rowSum) {
    rowSum.style.removeProperty('opacity');
    rowSum.style.removeProperty('visibility');
  }
  syncAttentionMatrixPostScoreMatrix('masked_raw');
  syncAttentionMatrixPostScoreRows(0);
  syncAttentionMatrixOutputRows(0);
  if (!slide) return;
  slide.classList.remove('attn23-show-postscore-mode');
  slide.classList.remove('attn23-show-postscore-centered');
  slide.classList.remove('attn23-show-postscore-scale');
  slide.classList.remove('attn23-show-postscore-softmax');
  slide.classList.remove('attn23-show-postscore-av');
  slide.classList.remove('attn23-show-postscore-output');
}

function resetAttentionMatrixScoreMaskVisuals() {
  const slide = document.getElementById('slide-23');
  state.attentionMatrix.maskProblemDone = false;
  state.attentionMatrix.maskAppliedDone = false;
  state.attentionMatrix.maskQuestionDone = false;
  state.attentionMatrix.maskVisibleCount = 0;
  state.attentionMatrix.maskMode = 'none';
  syncAttentionMatrixScoreMaskRows('none', 0);
  if (!slide) return;
  slide.classList.remove('attn23-show-score-mask-focus');
  slide.classList.remove('attn23-show-score-future-problem');
  slide.classList.remove('attn23-show-score-causal-mask');
  slide.classList.remove('attn23-show-score-mask-question');
}

function syncAttentionMatrixScoreMaskCellStyles() {
  const slide = document.getElementById('slide-23');
  if (!slide) return;
  const causalMask = slide.classList.contains('attn23-show-score-causal-mask');
  slide.querySelectorAll('.attn23-score-cell').forEach((cell) => {
    const row = cell.closest('.attn23-score-row');
    const value = cell.querySelector('.attn23-score-cell-value');
    const mask = cell.querySelector('.attn23-score-cell-mask');
    const future = cell.dataset.future === 'true';
    const rowMasked = !!(row && (
      row.classList.contains('is-mask-applied-visible')
      || row.classList.contains('is-mask-question-visible')
    ));
    const shouldMask = causalMask && future && rowMasked;
    if (value) value.classList.remove('is-dimmed');
    if (mask) mask.classList.toggle('is-visible', shouldMask);
  });
}

function syncAttentionMatrixPostScoreStageFromClasses() {
  const slide = document.getElementById('slide-23');
  if (!slide) return;

  const postScoreMode = slide.classList.contains('attn23-show-postscore-mode');
  if (!postScoreMode) {
    syncAttentionMatrixOutputRows(0);
    return;
  }

  syncAttentionMatrixPostScoreMatrix(state.attentionMatrix.postScoreMode || 'masked_raw');
  syncAttentionMatrixPostScoreRows(state.attentionMatrix.postScoreVisibleCount);
  syncAttentionMatrixOutputRows(state.attentionMatrix.outputVisibleCount);
}

function setAttentionMatrixScoreKState(transposed) {
  const label = document.getElementById('attn23-score-k-label');
  const dims = document.getElementById('attn23-score-k-dims');
  if (label) setMathHTML(label, inlineMath(transposed ? 'K^{\\mathsf{T}}' : 'K'));
  if (dims) setMathHTML(dims, inlineMath(transposed ? 'K^{\\mathsf{T}} \\in \\mathbb{R}^{d_k \\times S}' : 'K \\in \\mathbb{R}^{S \\times d_k}'));
}

function setAttentionMatrixProjectionOutputVisible(proj, visible) {
  const wrap = document.getElementById('attn23-proj-output-wrap-' + proj);
  if (!wrap) return;
  if (visible) {
    wrap.style.removeProperty('opacity');
    wrap.style.removeProperty('visibility');
    wrap.style.removeProperty('pointer-events');
    return;
  }
  wrap.style.setProperty('opacity', '0', 'important');
  wrap.style.setProperty('visibility', 'hidden', 'important');
  wrap.style.setProperty('pointer-events', 'none', 'important');
}

function syncAttentionMatrixScoreStageFromClasses() {
  const slide = document.getElementById('slide-23');
  if (!slide) return;

  const cleanup = slide.classList.contains('attn23-score-cleanup-active');
  const mode = slide.classList.contains('attn23-show-score-mode');
  const operands = slide.classList.contains('attn23-show-score-operands');
  const maskFocus = slide.classList.contains('attn23-show-score-mask-focus');
  const transposed = slide.classList.contains('attn23-show-score-transposed');
  const formula = slide.classList.contains('attn23-show-score-formula');
  const matrix = slide.classList.contains('attn23-show-score-matrix');

  const scoreStage = document.getElementById('attn23-score-stage');
  const sourceZone = slide.querySelector('.attn23-source-zone');
  const rail = document.getElementById('attn23-rail');
  const projOverlay = document.getElementById('attn23-proj-overlay');
  const qWrap = document.getElementById('attn23-score-q-wrap');
  const kWrap = document.getElementById('attn23-score-k-wrap');
  const sWrap = document.getElementById('attn23-score-s-wrap');
  const mul = document.getElementById('attn23-score-mul');
  const equals = document.getElementById('attn23-score-equals');
  const kRotate = document.getElementById('attn23-score-k-rotate');
  const kFrame = document.getElementById('attn23-score-k-body-frame');
  const kColHeaders = document.getElementById('attn23-score-k-col-headers');
  const kRowHeaders = document.getElementById('attn23-score-k-row-headers');

  if (scoreStage) {
    scoreStage.style.removeProperty('display');
    scoreStage.style.setProperty('opacity', mode ? '1' : '0', 'important');
    scoreStage.style.setProperty('visibility', mode ? 'visible' : 'hidden', 'important');
    scoreStage.style.setProperty('transform', mode ? 'translateY(0)' : 'translateY(0.24rem)', 'important');
  }

  [sourceZone, rail, projOverlay].forEach((el) => {
    if (!el) return;
    el.style.removeProperty('display');
    if (cleanup) {
      el.style.setProperty('opacity', '0', 'important');
      el.style.setProperty('visibility', 'hidden', 'important');
      el.style.setProperty('transform', 'translateY(-0.28rem) scale(0.985)', 'important');
      el.style.setProperty('pointer-events', 'none', 'important');
    } else {
      el.style.removeProperty('opacity');
      el.style.removeProperty('visibility');
      el.style.removeProperty('transform');
      el.style.removeProperty('pointer-events');
    }
  });

  [qWrap, kWrap].forEach((el) => {
    if (!el) return;
    el.style.removeProperty('display');
    el.style.setProperty('opacity', operands ? (maskFocus ? '0.76' : '1') : '0', 'important');
    el.style.setProperty('visibility', operands ? 'visible' : 'hidden', 'important');
    el.style.setProperty('transform', operands ? 'translateY(0)' : 'translateY(0.16rem)', 'important');
  });

  [mul, equals].forEach((el) => {
    if (!el) return;
    el.style.removeProperty('display');
    el.style.setProperty('opacity', formula ? '1' : '0', 'important');
    el.style.setProperty('visibility', formula ? 'visible' : 'hidden', 'important');
    el.style.setProperty('transform', formula ? 'translateY(0)' : 'translateY(0.1rem)', 'important');
  });

  if (sWrap) {
    sWrap.style.removeProperty('display');
    sWrap.style.setProperty('opacity', matrix ? '1' : '0', 'important');
    sWrap.style.setProperty('visibility', matrix ? 'visible' : 'hidden', 'important');
    sWrap.style.setProperty('transform', matrix ? 'translateY(0)' : 'translateY(0.16rem)', 'important');
  }

  if (kRotate) {
    kRotate.style.setProperty('transform', transposed ? 'rotate(90deg)' : 'rotate(0deg)', 'important');
  }
  if (kFrame) {
    kFrame.style.setProperty('width', transposed ? 'var(--attn23-score-k-transpose-frame-w)' : 'var(--attn23-score-operand-w)', 'important');
    kFrame.style.setProperty('height', transposed ? 'var(--attn23-score-k-transpose-frame-h)' : 'var(--attn23-score-operand-h)', 'important');
  }
  if (kColHeaders) {
    kColHeaders.style.setProperty('max-height', transposed ? '1rem' : '0', 'important');
    kColHeaders.style.setProperty('opacity', transposed ? '1' : '0', 'important');
    kColHeaders.style.setProperty('visibility', transposed ? 'visible' : 'hidden', 'important');
    kColHeaders.style.setProperty('transform', transposed ? 'translateY(0)' : 'translateY(0.08rem)', 'important');
  }
  if (kRowHeaders) {
    kRowHeaders.style.setProperty('opacity', transposed ? '0' : '1', 'important');
    kRowHeaders.style.setProperty('visibility', transposed ? 'hidden' : 'visible', 'important');
    kRowHeaders.style.setProperty('transform', transposed ? 'translateX(-0.08rem)' : 'translateX(0)', 'important');
  }
  syncAttentionMatrixScoreMaskCellStyles();
  syncAttentionMatrixPostScoreStageFromClasses();
}

function resetAttentionMatrixScoreVisuals() {
  const slide = document.getElementById('slide-23');
  clearAttentionMatrixScoreTimers();
  clearAttentionMatrixGhosts();
  state.attentionMatrix.scoreCenteredDone = false;
  state.attentionMatrix.scoreTransposedDone = false;
  state.attentionMatrix.scoreMatrixDone = false;
  state.attentionMatrix.scoreVisible = false;
  state.attentionMatrix.scoreVisibleCount = 0;
  syncAttentionMatrixScoreRows(0);
  resetAttentionMatrixScoreMaskVisuals();
  resetAttentionMatrixPostScoreVisuals();
  setAttentionMatrixScoreKState(false);
  if (!slide) return;
  ATTN_MATRIX_PROJS.forEach((proj) => setAttentionMatrixProjectionOutputVisible(proj, true));
  slide.classList.remove('attn23-score-cleanup-active');
  slide.classList.remove('attn23-show-score-mode');
  slide.classList.remove('attn23-show-score-operands');
  slide.classList.remove('attn23-show-score-transposed');
  slide.classList.remove('attn23-show-score-formula');
  slide.classList.remove('attn23-show-score-matrix');
  syncAttentionMatrixScoreStageFromClasses();
}

function resetAttentionMatrixVisuals() {
  const slide = document.getElementById('slide-23');
  clearAttentionMatrixTimers();
  clearAttentionMatrixProjectionTimers();
  clearAttentionMatrixScoreTimers();
  clearAttentionMatrixGhosts();
  state.attentionMatrix.tokenMatrixDone = false;
  state.attentionMatrix.xMatrixDone = false;
  state.attentionMatrix.tokenVisibleCount = 0;
  state.attentionMatrix.xVisibleCount = 0;
  state.attentionMatrix.projectionDone = false;
  state.attentionMatrix.projectionVisible = false;
  state.attentionMatrix.scoreCenteredDone = false;
  state.attentionMatrix.scoreTransposedDone = false;
  state.attentionMatrix.scoreMatrixDone = false;
  state.attentionMatrix.scoreVisible = false;
  state.attentionMatrix.scoreVisibleCount = 0;
  syncAttentionMatrixTokenSlots(0);
  syncAttentionMatrixXSlots(0);
  syncAttentionMatrixScoreRows(0);
  resetAttentionMatrixScoreMaskVisuals();
  resetAttentionMatrixPostScoreVisuals();
  if (!slide) return;
  ATTN_MATRIX_PROJS.forEach((proj) => setAttentionMatrixProjectionOutputVisible(proj, true));
  slide.classList.remove('attn23-show-token-matrix');
  slide.classList.remove('attn23-show-x-matrix');
  slide.classList.remove('attn23-token-settled');
  slide.classList.remove('attn23-full-settled');
  slide.classList.remove('attn23-show-projections');
  slide.classList.remove('attn23-show-copies');
  slide.classList.remove('attn23-show-proj-matrices');
  slide.classList.remove('attn23-show-proj-outputs');
  slide.classList.remove('attn23-score-cleanup-active');
  slide.classList.remove('attn23-show-score-mode');
  slide.classList.remove('attn23-show-score-operands');
  slide.classList.remove('attn23-show-score-transposed');
  slide.classList.remove('attn23-show-score-formula');
  slide.classList.remove('attn23-show-score-matrix');
  slide.querySelectorAll('.attn23-proj-col').forEach((col) => {
    col.classList.remove('is-output-visible');
  });
  syncAttentionMatrixScoreStageFromClasses();
}

function settleAttentionMatrixTokenState() {
  const slide = document.getElementById('slide-23');
  clearAttentionMatrixTimers();
  clearAttentionMatrixGhosts();
  clearAttentionMatrixProjectionTimers();
  clearAttentionMatrixScoreTimers();
  state.attentionMatrix.tokenMatrixDone = true;
  state.attentionMatrix.xMatrixDone = false;
  state.attentionMatrix.projectionDone = false;
  state.attentionMatrix.projectionVisible = false;
  state.attentionMatrix.scoreCenteredDone = false;
  state.attentionMatrix.scoreTransposedDone = false;
  state.attentionMatrix.scoreMatrixDone = false;
  state.attentionMatrix.scoreVisible = false;
  state.attentionMatrix.scoreVisibleCount = 0;
  state.attentionMatrix.tokenVisibleCount = ATTN_MATRIX_TOKENS.length;
  state.attentionMatrix.xVisibleCount = 0;
  syncAttentionMatrixTokenSlots(state.attentionMatrix.tokenVisibleCount);
  syncAttentionMatrixXSlots(0);
  syncAttentionMatrixScoreRows(0);
  resetAttentionMatrixScoreMaskVisuals();
  resetAttentionMatrixPostScoreVisuals();
  if (!slide) return;
  ATTN_MATRIX_PROJS.forEach((proj) => setAttentionMatrixProjectionOutputVisible(proj, true));
  slide.classList.add('attn23-show-token-matrix');
  slide.classList.add('attn23-token-settled');
  slide.classList.remove('attn23-show-x-matrix');
  slide.classList.remove('attn23-full-settled');
  slide.classList.remove('attn23-show-projections');
  slide.classList.remove('attn23-show-copies');
  slide.classList.remove('attn23-show-proj-matrices');
  slide.classList.remove('attn23-show-proj-outputs');
  slide.classList.remove('attn23-score-cleanup-active');
  slide.classList.remove('attn23-show-score-mode');
  slide.classList.remove('attn23-show-score-operands');
  slide.classList.remove('attn23-show-score-transposed');
  slide.classList.remove('attn23-show-score-formula');
  slide.classList.remove('attn23-show-score-matrix');
  slide.querySelectorAll('.attn23-proj-col').forEach((col) => {
    col.classList.remove('is-output-visible');
  });
  syncAttentionMatrixScoreStageFromClasses();
}

function settleAttentionMatrixFullState() {
  const slide = document.getElementById('slide-23');
  clearAttentionMatrixTimers();
  clearAttentionMatrixGhosts();
  clearAttentionMatrixProjectionTimers();
  clearAttentionMatrixScoreTimers();
  state.attentionMatrix.tokenMatrixDone = true;
  state.attentionMatrix.xMatrixDone = true;
  state.attentionMatrix.projectionDone = false;
  state.attentionMatrix.projectionVisible = false;
  state.attentionMatrix.scoreCenteredDone = false;
  state.attentionMatrix.scoreTransposedDone = false;
  state.attentionMatrix.scoreMatrixDone = false;
  state.attentionMatrix.scoreVisible = false;
  state.attentionMatrix.scoreVisibleCount = 0;
  state.attentionMatrix.tokenVisibleCount = ATTN_MATRIX_TOKENS.length;
  state.attentionMatrix.xVisibleCount = ATTN_MATRIX_TOKENS.length;
  syncAttentionMatrixTokenSlots(state.attentionMatrix.tokenVisibleCount);
  syncAttentionMatrixXSlots(state.attentionMatrix.xVisibleCount);
  syncAttentionMatrixScoreRows(0);
  resetAttentionMatrixScoreMaskVisuals();
  resetAttentionMatrixPostScoreVisuals();
  if (!slide) return;
  ATTN_MATRIX_PROJS.forEach((proj) => setAttentionMatrixProjectionOutputVisible(proj, true));
  slide.classList.add('attn23-show-token-matrix');
  slide.classList.add('attn23-show-x-matrix');
  slide.classList.add('attn23-token-settled');
  slide.classList.add('attn23-full-settled');
  slide.classList.remove('attn23-show-projections');
  slide.classList.remove('attn23-show-copies');
  slide.classList.remove('attn23-show-proj-matrices');
  slide.classList.remove('attn23-show-proj-outputs');
  slide.classList.remove('attn23-score-cleanup-active');
  slide.classList.remove('attn23-show-score-mode');
  slide.classList.remove('attn23-show-score-operands');
  slide.classList.remove('attn23-show-score-transposed');
  slide.classList.remove('attn23-show-score-formula');
  slide.classList.remove('attn23-show-score-matrix');
  slide.querySelectorAll('.attn23-proj-col').forEach((col) => {
    col.classList.remove('is-output-visible');
  });
  syncAttentionMatrixScoreStageFromClasses();
}

function updateAttentionMatrixProjectionOverlay() {
  const stage = document.getElementById('attn23-stage');
  const overlay = document.getElementById('attn23-proj-overlay');
  const xShell = document.getElementById('attn23-x-shell');
  const sourceLine = document.getElementById('attn23-proj-line-source');
  const busTrunkLine = document.getElementById('attn23-proj-line-bus-trunk');
  const busMainLine = document.getElementById('attn23-proj-line-bus-main');
  const copyNode = document.getElementById('attn23-proj-copy-node');
  const copyLabel = document.getElementById('attn23-proj-copy-label');
  if (!stage || !overlay || !xShell || !sourceLine || !busTrunkLine || !busMainLine || !copyNode || !copyLabel) return;

  const stageRect = stage.getBoundingClientRect();
  if (stageRect.width < 1 || stageRect.height < 1) return;
  overlay.setAttribute('viewBox', '0 0 ' + stageRect.width + ' ' + stageRect.height);

  const anchor = (el, fx, fy) => {
    const r = el.getBoundingClientRect();
    return {
      x: r.left - stageRect.left + r.width * fx,
      y: r.top - stageRect.top + r.height * fy
    };
  };

  const xBottom = anchor(xShell, 0.5, 1);
  const linePad = Math.max(stageRect.height * 0.009, 3.5);
  const branchTargets = [];
  ATTN_MATRIX_PROJS.forEach((proj) => {
    const mini = document.getElementById('attn23-mini-x-shell-' + proj);
    const matrix = document.getElementById('attn23-proj-matrix-shell-' + proj);
    const line = document.getElementById('attn23-proj-line-' + proj);
    if (!mini || !matrix || !line) return;
    branchTargets.push({
      line,
      target: anchor(mini, 0.5, 0),
      matrixTopY: anchor(matrix, 0.5, 0).y
    });
  });
  if (!branchTargets.length) return;

  const nearestCopyY = Math.min.apply(null, branchTargets.map((entry) => entry.target.y));
  const nearestMatrixY = Math.min.apply(null, branchTargets.map((entry) => entry.matrixTopY));
  const minCopyX = Math.min.apply(null, branchTargets.map((entry) => entry.target.x));
  const maxCopyX = Math.max.apply(null, branchTargets.map((entry) => entry.target.x));
  const nodeX = xBottom.x;
  const nodeRadius = Math.max(stageRect.height * 0.018, 6.4);
  const minNodeGap = Math.max(stageRect.height * 0.3, 84);
  const maxNodeY = nearestMatrixY - nodeRadius - Math.max(stageRect.height * 0.045, 9);
  const nodeY = Math.max(
    xBottom.y + nodeRadius + 2,
    Math.min(xBottom.y + minNodeGap, maxNodeY)
  );
  const busStartX = minCopyX;
  const busEndX = maxCopyX;
  const busClearBelowNode = Math.max(stageRect.height * 0.02, 6);
  const busMinY = nodeY + nodeRadius + busClearBelowNode;
  const busMaxY = nearestMatrixY - Math.max(stageRect.height * 0.026, 9);
  const busPreferredY = Math.min(
    nearestCopyY - Math.max(stageRect.height * 0.028, 11),
    nodeY + Math.max(stageRect.height * 0.072, 18)
  );
  let busY = Math.max(busMinY, Math.min(busPreferredY, busMaxY));
  if (!Number.isFinite(busY)) busY = busMinY;
  if (busY < busMinY) busY = busMinY;
  if (busY > busMaxY) busY = busMaxY;

  copyNode.setAttribute('cx', nodeX.toFixed(2));
  copyNode.setAttribute('cy', nodeY.toFixed(2));
  copyNode.setAttribute('r', nodeRadius.toFixed(2));
  copyLabel.setAttribute('x', nodeX.toFixed(2));
  copyLabel.setAttribute('y', nodeY.toFixed(2));

  const sourceY1 = xBottom.y + linePad;
  const sourceY2 = nodeY - nodeRadius - Math.max(stageRect.height * 0.006, 2);
  sourceLine.setAttribute('x1', nodeX.toFixed(2));
  sourceLine.setAttribute('y1', sourceY1.toFixed(2));
  sourceLine.setAttribute('x2', nodeX.toFixed(2));
  sourceLine.setAttribute('y2', sourceY2.toFixed(2));

  busTrunkLine.setAttribute('x1', nodeX.toFixed(2));
  busTrunkLine.setAttribute('y1', (nodeY + nodeRadius + Math.max(stageRect.height * 0.004, 1.2)).toFixed(2));
  busTrunkLine.setAttribute('x2', nodeX.toFixed(2));
  busTrunkLine.setAttribute('y2', busY.toFixed(2));

  busMainLine.setAttribute('x1', busStartX.toFixed(2));
  busMainLine.setAttribute('y1', busY.toFixed(2));
  busMainLine.setAttribute('x2', busEndX.toFixed(2));
  busMainLine.setAttribute('y2', busY.toFixed(2));

  branchTargets.forEach((entry) => {
    const toX = entry.target.x;
    const toY = entry.target.y - Math.max(stageRect.height * 0.004, 1.2);
    entry.line.setAttribute('x1', toX.toFixed(2));
    entry.line.setAttribute('y1', busY.toFixed(2));
    entry.line.setAttribute('x2', toX.toFixed(2));
    entry.line.setAttribute('y2', toY.toFixed(2));
  });
}

function resetAttentionMatrixProjectionVisuals() {
  const slide = document.getElementById('slide-23');
  clearAttentionMatrixProjectionTimers();
  state.attentionMatrix.projectionDone = false;
  state.attentionMatrix.projectionVisible = false;
  state.attentionMatrix.scoreCenteredDone = false;
  state.attentionMatrix.scoreTransposedDone = false;
  state.attentionMatrix.scoreMatrixDone = false;
  state.attentionMatrix.scoreVisible = false;
  state.attentionMatrix.scoreVisibleCount = 0;
  resetAttentionMatrixScoreMaskVisuals();
  resetAttentionMatrixPostScoreVisuals();
  if (!slide) return;
  ATTN_MATRIX_PROJS.forEach((proj) => setAttentionMatrixProjectionOutputVisible(proj, true));
  slide.classList.remove('attn23-show-projections');
  slide.classList.remove('attn23-show-copies');
  slide.classList.remove('attn23-show-proj-matrices');
  slide.classList.remove('attn23-show-proj-outputs');
  slide.classList.remove('attn23-score-cleanup-active');
  slide.classList.remove('attn23-show-score-mode');
  slide.classList.remove('attn23-show-score-operands');
  slide.classList.remove('attn23-show-score-transposed');
  slide.classList.remove('attn23-show-score-formula');
  slide.classList.remove('attn23-show-score-matrix');
  slide.querySelectorAll('.attn23-proj-col').forEach((col) => {
    col.classList.remove('is-output-visible');
  });
  syncAttentionMatrixScoreRows(0);
  syncAttentionMatrixScoreStageFromClasses();
}

function settleAttentionMatrixProjectionState() {
  const slide = document.getElementById('slide-23');
  settleAttentionMatrixFullState();
  clearAttentionMatrixProjectionTimers();
  state.attentionMatrix.projectionDone = true;
  state.attentionMatrix.projectionVisible = true;
  state.attentionMatrix.scoreCenteredDone = false;
  state.attentionMatrix.scoreTransposedDone = false;
  state.attentionMatrix.scoreMatrixDone = false;
  state.attentionMatrix.scoreVisible = false;
  state.attentionMatrix.scoreVisibleCount = 0;
  resetAttentionMatrixScoreMaskVisuals();
  resetAttentionMatrixPostScoreVisuals();
  if (!slide) return;
  ATTN_MATRIX_PROJS.forEach((proj) => setAttentionMatrixProjectionOutputVisible(proj, true));
  slide.classList.add('attn23-show-projections');
  slide.classList.add('attn23-show-copies');
  slide.classList.add('attn23-show-proj-matrices');
  slide.classList.add('attn23-show-proj-outputs');
  slide.classList.remove('attn23-score-cleanup-active');
  slide.classList.remove('attn23-show-score-mode');
  slide.classList.remove('attn23-show-score-operands');
  slide.classList.remove('attn23-show-score-transposed');
  slide.classList.remove('attn23-show-score-formula');
  slide.classList.remove('attn23-show-score-matrix');
  slide.querySelectorAll('.attn23-proj-col').forEach((col) => {
    col.classList.add('is-output-visible');
  });
  syncAttentionMatrixScoreRows(0);
  syncAttentionMatrixScoreStageFromClasses();
  updateAttentionMatrixProjectionOverlay();
}

function runAttentionMatrixProjectionSequence() {
  const slide = document.getElementById('slide-23');
  settleAttentionMatrixFullState();
  resetAttentionMatrixProjectionVisuals();
  if (!slide) return;
  slide.classList.add('attn23-show-projections');
  state.attentionMatrix.projectionVisible = true;
  updateAttentionMatrixProjectionOverlay();
  state.attentionMatrix.projectionRafIds.push(requestAnimationFrame(updateAttentionMatrixProjectionOverlay));

  const showCopiesTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 3) return;
    slide.classList.add('attn23-show-copies');
    updateAttentionMatrixProjectionOverlay();
  }, 40);
  state.attentionMatrix.projectionTimers.push(showCopiesTimer);

  const showMatricesTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 3) return;
    slide.classList.add('attn23-show-proj-matrices');
  }, 40 + ATTN_MATRIX_PROJ_ANIM_MS);
  state.attentionMatrix.projectionTimers.push(showMatricesTimer);

  ATTN_MATRIX_PROJS.forEach((proj, idx) => {
    const col = document.getElementById('attn23-proj-col-' + proj);
    if (!col) return;
    const outputTimer = setTimeout(() => {
      if (state.attentionMatrix.step < 3) return;
      slide.classList.add('attn23-show-proj-outputs');
      col.classList.add('is-output-visible');
      if (idx === ATTN_MATRIX_PROJS.length - 1) {
        state.attentionMatrix.projectionDone = true;
      }
    }, 40 + ATTN_MATRIX_PROJ_ANIM_MS + ATTN_MATRIX_PROJ_FADE_MS + (idx * ATTN_MATRIX_PROJ_STAGGER_MS));
    state.attentionMatrix.projectionTimers.push(outputTimer);
  });

  const settleTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 3) return;
    settleAttentionMatrixProjectionState();
  }, 40 + ATTN_MATRIX_PROJ_ANIM_MS + ATTN_MATRIX_PROJ_FADE_MS + ((ATTN_MATRIX_PROJS.length - 1) * ATTN_MATRIX_PROJ_STAGGER_MS) + ATTN_MATRIX_PROJ_ANIM_MS);
  state.attentionMatrix.projectionTimers.push(settleTimer);
}

function createAttentionMatrixProjectionGhost(proj) {
  const stage = document.getElementById('attn23-stage');
  const layer = document.getElementById('attn23-ghost-layer');
  const sourceShell = document.getElementById('attn23-proj-output-shell-' + proj);
  if (!stage || !layer || !sourceShell) return null;

  const stageRect = stage.getBoundingClientRect();
  const sourceRect = sourceShell.getBoundingClientRect();
  if (stageRect.width < 1 || stageRect.height < 1 || sourceRect.width < 1 || sourceRect.height < 1) return null;

  const ghost = cloneAttentionMatrixFragment(sourceShell, 'attn23-ghost-proj-output');
  if (!ghost) return null;
  ghost.dataset.proj = proj;
  placeAttentionMatrixGhost(ghost, sourceRect, stageRect);
  layer.appendChild(ghost);
  return ghost;
}

function createAttentionMatrixScoreTransportGhost(proj) {
  return createAttentionMatrixProjectionGhost(proj);
}

function createAttentionMatrixScoreToCenterGhost() {
  const stage = document.getElementById('attn23-stage');
  const layer = document.getElementById('attn23-ghost-layer');
  const sourceWrap = document.getElementById('attn23-score-s-wrap');
  if (!stage || !layer || !sourceWrap) return null;

  const stageRect = stage.getBoundingClientRect();
  const sourceRect = sourceWrap.getBoundingClientRect();
  if (stageRect.width < 1 || stageRect.height < 1 || sourceRect.width < 1 || sourceRect.height < 1) return null;

  const ghost = cloneAttentionMatrixFragment(sourceWrap, 'attn23-ghost-postscore-main');
  if (!ghost) return null;
  placeAttentionMatrixGhost(ghost, sourceRect, stageRect);
  layer.appendChild(ghost);
  return ghost;
}

function cloneAttentionMatrixFragment(node, extraClass) {
  if (!node) return null;
  const ghost = node.cloneNode(true);
  const stripIds = (el) => {
    if (!el || el.nodeType !== 1) return;
    el.removeAttribute('id');
    Array.from(el.children).forEach(stripIds);
  };
  stripIds(ghost);
  ghost.classList.add('attn23-ghost');
  if (extraClass) ghost.classList.add(extraClass);
  return ghost;
}

function placeAttentionMatrixGhost(ghost, rect, stageRect) {
  if (!ghost || !rect || !stageRect) return;
  ghost.style.left = (rect.left - stageRect.left).toFixed(2) + 'px';
  ghost.style.top = (rect.top - stageRect.top).toFixed(2) + 'px';
  ghost.style.width = rect.width.toFixed(2) + 'px';
  ghost.style.height = rect.height.toFixed(2) + 'px';
}

function animateAttentionMatrixGhost(ghost, sourceEl, targetEl, durationMs, options = {}) {
  const stage = document.getElementById('attn23-stage');
  if (!ghost || !stage || !sourceEl || !targetEl) return false;

  const stageRect = stage.getBoundingClientRect();
  const sourceRect = options.useGhostPosition ? ghost.getBoundingClientRect() : sourceEl.getBoundingClientRect();
  const targetRect = targetEl.getBoundingClientRect();
  if (
    stageRect.width < 1 || stageRect.height < 1
    || sourceRect.width < 1 || sourceRect.height < 1
    || targetRect.width < 1 || targetRect.height < 1
  ) return false;

  const fromX = sourceRect.left - stageRect.left + sourceRect.width * 0.5;
  const fromY = sourceRect.top - stageRect.top + sourceRect.height * 0.5;
  const toX = targetRect.left - stageRect.left + targetRect.width * 0.5;
  const toY = targetRect.top - stageRect.top + targetRect.height * 0.5;
  const dx = toX - fromX;
  const dy = toY - fromY;
  const scaleX = sourceRect.width > 0 ? targetRect.width / sourceRect.width : 1;
  const scaleY = sourceRect.height > 0 ? targetRect.height / sourceRect.height : 1;
  const rotateDeg = Number(options.rotateDeg) || 0;

  ghost.style.transition = 'transform ' + durationMs + 'ms cubic-bezier(0.2, 0.75, 0.3, 1), opacity ' + ATTN_MATRIX_FADE_MS + 'ms ease';
  const rafId = requestAnimationFrame(() => {
    ghost.style.transform = 'translate3d(' + dx.toFixed(2) + 'px, ' + dy.toFixed(2) + 'px, 0) rotate(' + rotateDeg.toFixed(2) + 'deg) scale(' + scaleX.toFixed(3) + ', ' + scaleY.toFixed(3) + ')';
  });
  const rafStore = Array.isArray(options.rafStore) ? options.rafStore : state.attentionMatrix.rafIds;
  rafStore.push(rafId);
  return true;
}

function runAttentionMatrixSharedElementMove(ghost, sourceEl, targetEl, durationMs, options = {}) {
  return animateAttentionMatrixGhost(ghost, sourceEl, targetEl, durationMs, options);
}

function createAttentionMatrixTokenGhost(token) {
  const stage = document.getElementById('attn23-stage');
  const layer = document.getElementById('attn23-ghost-layer');
  const sourceWrap = document.getElementById('attn23-source-chip-wrap-' + token);
  if (!stage || !layer || !sourceWrap) return null;

  const stageRect = stage.getBoundingClientRect();
  const sourceRect = sourceWrap.getBoundingClientRect();
  if (stageRect.width < 1 || stageRect.height < 1 || sourceRect.width < 1 || sourceRect.height < 1) return null;

  const ghost = cloneAttentionMatrixFragment(sourceWrap, 'attn23-ghost-token');
  if (!ghost) return null;
  placeAttentionMatrixGhost(ghost, sourceRect, stageRect);
  layer.appendChild(ghost);
  return ghost;
}

function createAttentionMatrixXGhost(token) {
  const stage = document.getElementById('attn23-stage');
  const layer = document.getElementById('attn23-ghost-layer');
  const sourceWrap = document.getElementById('attn23-source-vector-wrap-' + token);
  if (!stage || !layer || !sourceWrap) return null;

  const stageRect = stage.getBoundingClientRect();
  const sourceRect = sourceWrap.getBoundingClientRect();
  if (stageRect.width < 1 || stageRect.height < 1 || sourceRect.width < 1 || sourceRect.height < 1) return null;

  const ghost = cloneAttentionMatrixFragment(sourceWrap, 'attn23-ghost-x');
  if (!ghost) return null;
  placeAttentionMatrixGhost(ghost, sourceRect, stageRect);
  layer.appendChild(ghost);
  return ghost;
}

function runAttentionMatrixScoreCleanup() {
  const slide = document.getElementById('slide-23');
  if (!slide) return;
  slide.classList.add('attn23-score-cleanup-active');
  slide.classList.remove('attn23-show-score-mode');
  slide.classList.remove('attn23-show-score-operands');
  slide.classList.remove('attn23-show-score-transposed');
  slide.classList.remove('attn23-show-score-formula');
  slide.classList.remove('attn23-show-score-matrix');
  state.attentionMatrix.scoreVisible = false;
  syncAttentionMatrixScoreStageFromClasses();
}

function settleAttentionMatrixScoreCenterState() {
  const slide = document.getElementById('slide-23');
  settleAttentionMatrixProjectionState();
  clearAttentionMatrixScoreTimers();
  clearAttentionMatrixGhosts();
  state.attentionMatrix.scoreCenteredDone = true;
  state.attentionMatrix.scoreTransposedDone = false;
  state.attentionMatrix.scoreMatrixDone = false;
  state.attentionMatrix.scoreVisible = true;
  state.attentionMatrix.scoreVisibleCount = 0;
  syncAttentionMatrixScoreRows(0);
  resetAttentionMatrixScoreMaskVisuals();
  resetAttentionMatrixPostScoreVisuals();
  setAttentionMatrixScoreKState(false);
  if (!slide) return;
  setAttentionMatrixProjectionOutputVisible('q', false);
  setAttentionMatrixProjectionOutputVisible('k', false);
  slide.classList.add('attn23-score-cleanup-active');
  slide.classList.add('attn23-show-score-mode');
  slide.classList.add('attn23-show-score-operands');
  slide.classList.remove('attn23-show-score-transposed');
  slide.classList.remove('attn23-show-score-formula');
  slide.classList.remove('attn23-show-score-matrix');
  syncAttentionMatrixScoreStageFromClasses();
}

function settleAttentionMatrixScoreTransposeState() {
  const slide = document.getElementById('slide-23');
  clearAttentionMatrixScoreTimers();
  clearAttentionMatrixGhosts();
  state.attentionMatrix.scoreCenteredDone = true;
  state.attentionMatrix.scoreTransposedDone = true;
  state.attentionMatrix.scoreMatrixDone = false;
  state.attentionMatrix.scoreVisible = true;
  state.attentionMatrix.scoreVisibleCount = 0;
  syncAttentionMatrixScoreRows(0);
  resetAttentionMatrixScoreMaskVisuals();
  resetAttentionMatrixPostScoreVisuals();
  setAttentionMatrixScoreKState(true);
  if (!slide) return;
  setAttentionMatrixProjectionOutputVisible('q', false);
  setAttentionMatrixProjectionOutputVisible('k', false);
  slide.classList.add('attn23-score-cleanup-active');
  slide.classList.add('attn23-show-score-mode');
  slide.classList.add('attn23-show-score-operands');
  slide.classList.add('attn23-show-score-transposed');
  slide.classList.remove('attn23-show-score-formula');
  slide.classList.remove('attn23-show-score-matrix');
  syncAttentionMatrixScoreStageFromClasses();
}

function settleAttentionMatrixScoreMatrixState() {
  const slide = document.getElementById('slide-23');
  clearAttentionMatrixScoreTimers();
  clearAttentionMatrixGhosts();
  state.attentionMatrix.scoreCenteredDone = true;
  state.attentionMatrix.scoreTransposedDone = true;
  state.attentionMatrix.scoreMatrixDone = true;
  state.attentionMatrix.scoreVisible = true;
  state.attentionMatrix.scoreVisibleCount = ATTN_MATRIX_SCORE_TOKENS.length;
  syncAttentionMatrixScoreRows(state.attentionMatrix.scoreVisibleCount);
  resetAttentionMatrixScoreMaskVisuals();
  resetAttentionMatrixPostScoreVisuals();
  setAttentionMatrixScoreKState(true);
  if (!slide) return;
  setAttentionMatrixProjectionOutputVisible('q', false);
  setAttentionMatrixProjectionOutputVisible('k', false);
  slide.classList.add('attn23-score-cleanup-active');
  slide.classList.add('attn23-show-score-mode');
  slide.classList.add('attn23-show-score-operands');
  slide.classList.add('attn23-show-score-transposed');
  slide.classList.add('attn23-show-score-formula');
  slide.classList.add('attn23-show-score-matrix');
  syncAttentionMatrixScoreStageFromClasses();
}

function runAttentionMatrixScoreCenterSequence() {
  const slide = document.getElementById('slide-23');
  if (!slide) return;

  settleAttentionMatrixProjectionState();
  clearAttentionMatrixScoreTimers();
  clearAttentionMatrixGhosts();
  state.attentionMatrix.scoreCenteredDone = false;
  state.attentionMatrix.scoreTransposedDone = false;
  state.attentionMatrix.scoreMatrixDone = false;
  state.attentionMatrix.scoreVisible = false;
  state.attentionMatrix.scoreVisibleCount = 0;
  syncAttentionMatrixScoreRows(0);
  setAttentionMatrixScoreKState(false);
  setAttentionMatrixProjectionOutputVisible('q', true);
  setAttentionMatrixProjectionOutputVisible('k', true);
  slide.classList.remove('attn23-score-cleanup-active');
  slide.classList.remove('attn23-show-score-mode');
  slide.classList.remove('attn23-show-score-operands');
  slide.classList.remove('attn23-show-score-transposed');
  slide.classList.remove('attn23-show-score-formula');
  slide.classList.remove('attn23-show-score-matrix');
  syncAttentionMatrixScoreStageFromClasses();

  const qSource = document.getElementById('attn23-proj-output-shell-q');
  const kSource = document.getElementById('attn23-proj-output-shell-k');
  const qTarget = document.getElementById('attn23-score-q-shell');
  const kTarget = document.getElementById('attn23-score-k-shell');
  const qGhost = createAttentionMatrixScoreTransportGhost('q');
  const kGhost = createAttentionMatrixScoreTransportGhost('k');
  if (!qGhost || !kGhost || !qSource || !kSource || !qTarget || !kTarget) {
    if (qGhost && qGhost.parentNode) qGhost.parentNode.removeChild(qGhost);
    if (kGhost && kGhost.parentNode) kGhost.parentNode.removeChild(kGhost);
    settleAttentionMatrixScoreCenterState();
    return;
  }

  runAttentionMatrixScoreCleanup();
  setAttentionMatrixProjectionOutputVisible('q', false);
  setAttentionMatrixProjectionOutputVisible('k', false);

  const liftDelay = 24;
  const moveTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 4) return;
    const qMoved = runAttentionMatrixSharedElementMove(
      qGhost,
      qSource,
      qTarget,
      ATTN_MATRIX_SCORE_CENTER_MS,
      { rafStore: state.attentionMatrix.scoreRafIds, useGhostPosition: true }
    );
    const kMoved = runAttentionMatrixSharedElementMove(
      kGhost,
      kSource,
      kTarget,
      ATTN_MATRIX_SCORE_CENTER_MS,
      { rafStore: state.attentionMatrix.scoreRafIds, useGhostPosition: true }
    );
    if (!qMoved || !kMoved) {
      settleAttentionMatrixScoreCenterState();
    }
  }, liftDelay);
  state.attentionMatrix.scoreTimers.push(moveTimer);

  const operandTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 4) return;
    slide.classList.add('attn23-show-score-mode');
    slide.classList.add('attn23-show-score-operands');
    state.attentionMatrix.scoreVisible = true;
    syncAttentionMatrixScoreStageFromClasses();
    qGhost.style.opacity = '0';
    kGhost.style.opacity = '0';
  }, liftDelay + ATTN_MATRIX_SCORE_CENTER_MS - 90);
  state.attentionMatrix.scoreTimers.push(operandTimer);

  const cleanupGhostTimer = setTimeout(() => {
    if (qGhost && qGhost.parentNode) qGhost.parentNode.removeChild(qGhost);
    if (kGhost && kGhost.parentNode) kGhost.parentNode.removeChild(kGhost);
  }, liftDelay + ATTN_MATRIX_SCORE_CENTER_MS + ATTN_MATRIX_FADE_MS + 60);
  state.attentionMatrix.scoreTimers.push(cleanupGhostTimer);

  const settleTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 4) return;
    settleAttentionMatrixScoreCenterState();
  }, liftDelay + ATTN_MATRIX_SCORE_CENTER_MS + ATTN_MATRIX_FADE_MS + 80);
  state.attentionMatrix.scoreTimers.push(settleTimer);
}

function runAttentionMatrixScoreTransposeSequence() {
  const slide = document.getElementById('slide-23');
  if (!slide) return;

  settleAttentionMatrixScoreCenterState();
  clearAttentionMatrixScoreTimers();
  state.attentionMatrix.scoreTransposedDone = false;
  setAttentionMatrixScoreKState(false);

  const transposeTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 5) return;
    slide.classList.add('attn23-show-score-transposed');
    setAttentionMatrixScoreKState(true);
    syncAttentionMatrixScoreStageFromClasses();
  }, 30);
  state.attentionMatrix.scoreTimers.push(transposeTimer);

  const settleTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 5) return;
    settleAttentionMatrixScoreTransposeState();
  }, 30 + ATTN_MATRIX_SCORE_TRANSPOSE_MS);
  state.attentionMatrix.scoreTimers.push(settleTimer);
}

function runAttentionMatrixScoreMatrixSequence() {
  const slide = document.getElementById('slide-23');
  if (!slide) return;

  settleAttentionMatrixScoreTransposeState();
  clearAttentionMatrixScoreTimers();
  state.attentionMatrix.scoreMatrixDone = false;
  state.attentionMatrix.scoreVisibleCount = 0;
  syncAttentionMatrixScoreRows(0);
  slide.classList.remove('attn23-show-score-formula');
  slide.classList.remove('attn23-show-score-matrix');
  syncAttentionMatrixScoreStageFromClasses();

  const formulaTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 6) return;
    slide.classList.add('attn23-show-score-formula');
    syncAttentionMatrixScoreStageFromClasses();
  }, 40);
  state.attentionMatrix.scoreTimers.push(formulaTimer);

  const matrixTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 6) return;
    slide.classList.add('attn23-show-score-matrix');
    syncAttentionMatrixScoreStageFromClasses();
  }, 40 + ATTN_MATRIX_SCORE_FADE_MS);
  state.attentionMatrix.scoreTimers.push(matrixTimer);

  ATTN_MATRIX_SCORE_TOKENS.forEach((token, idx) => {
    const rowTimer = setTimeout(() => {
      if (state.attentionMatrix.step < 6) return;
      state.attentionMatrix.scoreVisibleCount = Math.max(state.attentionMatrix.scoreVisibleCount, idx + 1);
      syncAttentionMatrixScoreRows(state.attentionMatrix.scoreVisibleCount);
    }, 40 + ATTN_MATRIX_SCORE_FADE_MS + 70 + (idx * ATTN_MATRIX_SCORE_ROW_STAGGER_MS));
    state.attentionMatrix.scoreTimers.push(rowTimer);
  });

  const settleTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 6) return;
    settleAttentionMatrixScoreMatrixState();
  }, 40 + ATTN_MATRIX_SCORE_FADE_MS + 70 + ((ATTN_MATRIX_SCORE_TOKENS.length - 1) * ATTN_MATRIX_SCORE_ROW_STAGGER_MS) + ATTN_MATRIX_SCORE_FADE_MS);
  state.attentionMatrix.scoreTimers.push(settleTimer);
}

function settleAttentionMatrixMaskProblemState() {
  const slide = document.getElementById('slide-23');
  settleAttentionMatrixScoreMatrixState();
  clearAttentionMatrixScoreTimers();
  clearAttentionMatrixGhosts();
  state.attentionMatrix.maskProblemDone = true;
  state.attentionMatrix.maskAppliedDone = false;
  state.attentionMatrix.maskQuestionDone = false;
  state.attentionMatrix.maskVisibleCount = ATTN_MATRIX_SCORE_TOKENS.length;
  state.attentionMatrix.maskMode = 'problem';
  resetAttentionMatrixPostScoreVisuals();
  if (!slide) return;
  slide.classList.add('attn23-show-score-mask-focus');
  slide.classList.add('attn23-show-score-future-problem');
  slide.classList.remove('attn23-show-score-causal-mask');
  slide.classList.remove('attn23-show-score-mask-question');
  syncAttentionMatrixScoreMaskRows('problem', state.attentionMatrix.maskVisibleCount);
  syncAttentionMatrixScoreStageFromClasses();
}

function settleAttentionMatrixCausalMaskState() {
  const slide = document.getElementById('slide-23');
  settleAttentionMatrixScoreMatrixState();
  clearAttentionMatrixScoreTimers();
  clearAttentionMatrixGhosts();
  state.attentionMatrix.maskProblemDone = true;
  state.attentionMatrix.maskAppliedDone = true;
  state.attentionMatrix.maskQuestionDone = false;
  state.attentionMatrix.maskVisibleCount = ATTN_MATRIX_SCORE_TOKENS.length;
  state.attentionMatrix.maskMode = 'mask';
  resetAttentionMatrixPostScoreVisuals();
  if (!slide) return;
  slide.classList.add('attn23-show-score-mask-focus');
  slide.classList.remove('attn23-show-score-future-problem');
  slide.classList.add('attn23-show-score-causal-mask');
  slide.classList.remove('attn23-show-score-mask-question');
  syncAttentionMatrixScoreMaskRows('mask', state.attentionMatrix.maskVisibleCount);
  syncAttentionMatrixScoreStageFromClasses();
}

function settleAttentionMatrixMaskQuestionState() {
  const slide = document.getElementById('slide-23');
  settleAttentionMatrixCausalMaskState();
  clearAttentionMatrixScoreTimers();
  clearAttentionMatrixGhosts();
  state.attentionMatrix.maskProblemDone = true;
  state.attentionMatrix.maskAppliedDone = true;
  state.attentionMatrix.maskQuestionDone = true;
  state.attentionMatrix.maskVisibleCount = ATTN_MATRIX_SCORE_TOKENS.length;
  state.attentionMatrix.maskMode = 'question';
  resetAttentionMatrixPostScoreVisuals();
  if (!slide) return;
  slide.classList.add('attn23-show-score-mask-focus');
  slide.classList.add('attn23-show-score-causal-mask');
  slide.classList.add('attn23-show-score-mask-question');
  syncAttentionMatrixScoreMaskRows('question', state.attentionMatrix.maskVisibleCount);
  syncAttentionMatrixScoreStageFromClasses();
}

function runAttentionMatrixMaskProblemSequence() {
  const slide = document.getElementById('slide-23');
  if (!slide) return;

  settleAttentionMatrixScoreMatrixState();
  clearAttentionMatrixScoreTimers();
  clearAttentionMatrixGhosts();
  resetAttentionMatrixScoreMaskVisuals();
  state.attentionMatrix.maskProblemDone = false;
  state.attentionMatrix.maskAppliedDone = false;
  state.attentionMatrix.maskQuestionDone = false;
  state.attentionMatrix.maskVisibleCount = 0;
  state.attentionMatrix.maskMode = 'problem';
  slide.classList.add('attn23-show-score-mask-focus');
  slide.classList.add('attn23-show-score-future-problem');
  syncAttentionMatrixScoreMaskRows('problem', 0);
  syncAttentionMatrixScoreStageFromClasses();

  ATTN_MATRIX_SCORE_TOKENS.forEach((token, idx) => {
    const timer = setTimeout(() => {
      if (state.attentionMatrix.step < 7) return;
      state.attentionMatrix.maskVisibleCount = Math.max(state.attentionMatrix.maskVisibleCount, idx + 1);
      syncAttentionMatrixScoreMaskRows('problem', state.attentionMatrix.maskVisibleCount);
    }, 40 + (idx * ATTN_MATRIX_MASK_ROW_STAGGER_MS));
    state.attentionMatrix.scoreTimers.push(timer);
  });

  const settleTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 7) return;
    settleAttentionMatrixMaskProblemState();
  }, 40 + ((ATTN_MATRIX_SCORE_TOKENS.length - 1) * ATTN_MATRIX_MASK_ROW_STAGGER_MS) + ATTN_MATRIX_MASK_FADE_MS);
  state.attentionMatrix.scoreTimers.push(settleTimer);
}

function runAttentionMatrixCausalMaskSequence() {
  const slide = document.getElementById('slide-23');
  if (!slide) return;

  settleAttentionMatrixMaskProblemState();
  clearAttentionMatrixScoreTimers();
  state.attentionMatrix.maskAppliedDone = false;
  state.attentionMatrix.maskQuestionDone = false;
  state.attentionMatrix.maskVisibleCount = 0;
  state.attentionMatrix.maskMode = 'mask';
  slide.classList.add('attn23-show-score-mask-focus');
  slide.classList.remove('attn23-show-score-future-problem');
  slide.classList.add('attn23-show-score-causal-mask');
  slide.classList.remove('attn23-show-score-mask-question');
  syncAttentionMatrixScoreMaskRows('mask', 0);

  ATTN_MATRIX_SCORE_TOKENS.forEach((token, idx) => {
    const timer = setTimeout(() => {
      if (state.attentionMatrix.step < 8) return;
      state.attentionMatrix.maskVisibleCount = Math.max(state.attentionMatrix.maskVisibleCount, idx + 1);
      syncAttentionMatrixScoreMaskRows('mask', state.attentionMatrix.maskVisibleCount);
    }, 40 + (idx * ATTN_MATRIX_MASK_ROW_STAGGER_MS));
    state.attentionMatrix.scoreTimers.push(timer);
  });

  const settleTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 8) return;
    settleAttentionMatrixCausalMaskState();
  }, 40 + ((ATTN_MATRIX_SCORE_TOKENS.length - 1) * ATTN_MATRIX_MASK_ROW_STAGGER_MS) + ATTN_MATRIX_MASK_FADE_MS);
  state.attentionMatrix.scoreTimers.push(settleTimer);
}

function runAttentionMatrixMaskQuestionSequence() {
  const slide = document.getElementById('slide-23');
  if (!slide) return;

  settleAttentionMatrixCausalMaskState();
  clearAttentionMatrixScoreTimers();
  state.attentionMatrix.maskQuestionDone = false;
  state.attentionMatrix.maskVisibleCount = 0;
  state.attentionMatrix.maskMode = 'question';
  slide.classList.add('attn23-show-score-mask-question');
  syncAttentionMatrixScoreMaskCellStyles();
  syncAttentionMatrixScoreCellText('question');

  ATTN_MATRIX_SCORE_TOKENS.forEach((token, idx) => {
    const timer = setTimeout(() => {
      if (state.attentionMatrix.step < 9) return;
      state.attentionMatrix.maskVisibleCount = Math.max(state.attentionMatrix.maskVisibleCount, idx + 1);
      setAttentionMatrixScoreMaskRowState(token, 'question');
      syncAttentionMatrixScoreMaskCellStyles();
      syncAttentionMatrixScoreCellText('question');
    }, 40 + (idx * ATTN_MATRIX_MASK_ROW_STAGGER_MS));
    state.attentionMatrix.scoreTimers.push(timer);
  });

  const settleTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 9) return;
    settleAttentionMatrixMaskQuestionState();
  }, 40 + ((ATTN_MATRIX_SCORE_TOKENS.length - 1) * ATTN_MATRIX_MASK_ROW_STAGGER_MS) + ATTN_MATRIX_MASK_FADE_MS);
  state.attentionMatrix.scoreTimers.push(settleTimer);
}

function clearAttentionMatrixPostScoreRowActivity() {
  ATTN_MATRIX_SCORE_TOKENS.forEach((token) => {
    const row = document.getElementById('attn23-postscore-row-' + token);
    if (row) row.classList.remove('is-softmax-active', 'is-output-active');
  });
  const vWrap = document.getElementById('attn23-postscore-v-wrap');
  if (vWrap) vWrap.classList.remove('is-source-emphasis');
}

function settleAttentionMatrixPostScoreCenterState() {
  const slide = document.getElementById('slide-23');
  settleAttentionMatrixMaskQuestionState();
  clearAttentionMatrixScoreTimers();
  clearAttentionMatrixGhosts();
  state.attentionMatrix.postScoreCenteredDone = true;
  state.attentionMatrix.scaledMatrixDone = false;
  state.attentionMatrix.attentionMatrixDone = false;
  state.attentionMatrix.valueMatrixVisibleDone = false;
  state.attentionMatrix.outputMatrixDone = false;
  state.attentionMatrix.postScoreMode = 'masked_raw';
  state.attentionMatrix.postScoreVisibleCount = 0;
  state.attentionMatrix.outputVisibleCount = 0;
  clearAttentionMatrixPostScoreRowActivity();
  syncAttentionMatrixPostScoreMatrix('masked_raw');
  syncAttentionMatrixPostScoreRows(0);
  syncAttentionMatrixOutputRows(0);
  if (!slide) return;
  slide.classList.add('attn23-score-cleanup-active');
  slide.classList.add('attn23-show-score-mode');
  slide.classList.add('attn23-show-postscore-mode');
  slide.classList.add('attn23-show-postscore-centered');
  slide.classList.remove('attn23-show-score-operands');
  slide.classList.remove('attn23-show-score-transposed');
  slide.classList.remove('attn23-show-score-formula');
  slide.classList.remove('attn23-show-score-matrix');
  slide.classList.remove('attn23-show-score-mask-focus');
  slide.classList.remove('attn23-show-score-future-problem');
  slide.classList.remove('attn23-show-score-causal-mask');
  slide.classList.remove('attn23-show-score-mask-question');
  slide.classList.remove('attn23-show-postscore-scale');
  slide.classList.remove('attn23-show-postscore-softmax');
  slide.classList.remove('attn23-show-postscore-av');
  slide.classList.remove('attn23-show-postscore-output');
  syncAttentionMatrixScoreStageFromClasses();
}

function settleAttentionMatrixScaleState() {
  const slide = document.getElementById('slide-23');
  settleAttentionMatrixPostScoreCenterState();
  clearAttentionMatrixScoreTimers();
  clearAttentionMatrixGhosts();
  state.attentionMatrix.postScoreCenteredDone = true;
  state.attentionMatrix.scaledMatrixDone = true;
  state.attentionMatrix.attentionMatrixDone = false;
  state.attentionMatrix.valueMatrixVisibleDone = false;
  state.attentionMatrix.outputMatrixDone = false;
  state.attentionMatrix.postScoreMode = 'scaled';
  state.attentionMatrix.postScoreVisibleCount = 0;
  state.attentionMatrix.outputVisibleCount = 0;
  clearAttentionMatrixPostScoreRowActivity();
  syncAttentionMatrixPostScoreMatrix('scaled');
  syncAttentionMatrixPostScoreRows(0);
  syncAttentionMatrixOutputRows(0);
  if (!slide) return;
  slide.classList.add('attn23-score-cleanup-active');
  slide.classList.add('attn23-show-score-mode');
  slide.classList.add('attn23-show-postscore-mode');
  slide.classList.add('attn23-show-postscore-centered');
  slide.classList.add('attn23-show-postscore-scale');
  slide.classList.remove('attn23-show-postscore-softmax');
  slide.classList.remove('attn23-show-postscore-av');
  slide.classList.remove('attn23-show-postscore-output');
  syncAttentionMatrixScoreStageFromClasses();
}

function settleAttentionMatrixSoftmaxState() {
  const slide = document.getElementById('slide-23');
  settleAttentionMatrixScaleState();
  clearAttentionMatrixScoreTimers();
  clearAttentionMatrixGhosts();
  state.attentionMatrix.postScoreCenteredDone = true;
  state.attentionMatrix.scaledMatrixDone = true;
  state.attentionMatrix.attentionMatrixDone = true;
  state.attentionMatrix.valueMatrixVisibleDone = false;
  state.attentionMatrix.outputMatrixDone = false;
  state.attentionMatrix.postScoreMode = 'attention';
  state.attentionMatrix.postScoreVisibleCount = ATTN_MATRIX_SCORE_TOKENS.length;
  state.attentionMatrix.outputVisibleCount = 0;
  clearAttentionMatrixPostScoreRowActivity();
  syncAttentionMatrixPostScoreMatrix('attention');
  syncAttentionMatrixPostScoreRows(state.attentionMatrix.postScoreVisibleCount);
  syncAttentionMatrixOutputRows(0);
  if (!slide) return;
  slide.classList.add('attn23-score-cleanup-active');
  slide.classList.add('attn23-show-score-mode');
  slide.classList.add('attn23-show-postscore-mode');
  slide.classList.add('attn23-show-postscore-centered');
  slide.classList.add('attn23-show-postscore-softmax');
  slide.classList.remove('attn23-show-postscore-scale');
  slide.classList.remove('attn23-show-postscore-av');
  slide.classList.remove('attn23-show-postscore-output');
  const rowSum = document.getElementById('attn23-postscore-row-sum');
  if (rowSum) {
    rowSum.style.removeProperty('opacity');
    rowSum.style.removeProperty('visibility');
  }
  syncAttentionMatrixScoreStageFromClasses();
}

function settleAttentionMatrixValueEntryState() {
  const slide = document.getElementById('slide-23');
  settleAttentionMatrixSoftmaxState();
  clearAttentionMatrixScoreTimers();
  clearAttentionMatrixGhosts();
  state.attentionMatrix.postScoreCenteredDone = true;
  state.attentionMatrix.scaledMatrixDone = true;
  state.attentionMatrix.attentionMatrixDone = true;
  state.attentionMatrix.valueMatrixVisibleDone = true;
  state.attentionMatrix.outputMatrixDone = false;
  state.attentionMatrix.postScoreMode = 'attention';
  state.attentionMatrix.postScoreVisibleCount = ATTN_MATRIX_SCORE_TOKENS.length;
  state.attentionMatrix.outputVisibleCount = 0;
  clearAttentionMatrixPostScoreRowActivity();
  syncAttentionMatrixPostScoreMatrix('attention');
  syncAttentionMatrixPostScoreRows(state.attentionMatrix.postScoreVisibleCount);
  syncAttentionMatrixOutputRows(0);
  if (!slide) return;
  slide.classList.add('attn23-score-cleanup-active');
  slide.classList.add('attn23-show-score-mode');
  slide.classList.add('attn23-show-postscore-mode');
  slide.classList.add('attn23-show-postscore-av');
  slide.classList.remove('attn23-show-postscore-centered');
  slide.classList.remove('attn23-show-postscore-scale');
  slide.classList.remove('attn23-show-postscore-softmax');
  slide.classList.remove('attn23-show-postscore-output');
  syncAttentionMatrixScoreStageFromClasses();
}

function settleAttentionMatrixOutputState() {
  const slide = document.getElementById('slide-23');
  settleAttentionMatrixValueEntryState();
  clearAttentionMatrixScoreTimers();
  clearAttentionMatrixGhosts();
  state.attentionMatrix.postScoreCenteredDone = true;
  state.attentionMatrix.scaledMatrixDone = true;
  state.attentionMatrix.attentionMatrixDone = true;
  state.attentionMatrix.valueMatrixVisibleDone = true;
  state.attentionMatrix.outputMatrixDone = true;
  state.attentionMatrix.postScoreMode = 'attention';
  state.attentionMatrix.postScoreVisibleCount = ATTN_MATRIX_SCORE_TOKENS.length;
  state.attentionMatrix.outputVisibleCount = ATTN_MATRIX_TOKENS.length;
  clearAttentionMatrixPostScoreRowActivity();
  syncAttentionMatrixPostScoreMatrix('attention');
  syncAttentionMatrixPostScoreRows(state.attentionMatrix.postScoreVisibleCount);
  syncAttentionMatrixOutputRows(state.attentionMatrix.outputVisibleCount);
  if (!slide) return;
  slide.classList.add('attn23-show-postscore-output');
  syncAttentionMatrixScoreStageFromClasses();
}

function runAttentionMatrixPostScoreCenterSequence() {
  const slide = document.getElementById('slide-23');
  if (!slide) return;

  settleAttentionMatrixMaskQuestionState();
  clearAttentionMatrixScoreTimers();
  clearAttentionMatrixGhosts();
  state.attentionMatrix.postScoreCenteredDone = false;
  state.attentionMatrix.scaledMatrixDone = false;
  state.attentionMatrix.attentionMatrixDone = false;
  state.attentionMatrix.valueMatrixVisibleDone = false;
  state.attentionMatrix.outputMatrixDone = false;
  state.attentionMatrix.postScoreMode = 'masked_raw';
  state.attentionMatrix.postScoreVisibleCount = 0;
  state.attentionMatrix.outputVisibleCount = 0;
  clearAttentionMatrixPostScoreRowActivity();
  syncAttentionMatrixOutputRows(0);

  const sourceWrap = document.getElementById('attn23-score-s-wrap');
  const targetWrap = document.getElementById('attn23-postscore-main-wrap');
  const ghost = createAttentionMatrixScoreToCenterGhost();
  if (!ghost || !sourceWrap || !targetWrap) {
    if (ghost && ghost.parentNode) ghost.parentNode.removeChild(ghost);
    settleAttentionMatrixPostScoreCenterState();
    return;
  }

  slide.classList.add('attn23-score-cleanup-active');
  slide.classList.add('attn23-show-score-mode');
  slide.classList.add('attn23-show-postscore-mode');
  slide.classList.remove('attn23-show-postscore-centered');
  slide.classList.remove('attn23-show-postscore-scale');
  slide.classList.remove('attn23-show-postscore-softmax');
  slide.classList.remove('attn23-show-postscore-av');
  slide.classList.remove('attn23-show-postscore-output');

  const liftDelay = 24;
  const moveTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 10) return;
    const moved = runAttentionMatrixSharedElementMove(
      ghost,
      sourceWrap,
      targetWrap,
      ATTN_MATRIX_POSTSCORE_CENTER_MS,
      { rafStore: state.attentionMatrix.scoreRafIds, useGhostPosition: true }
    );
    if (!moved) settleAttentionMatrixPostScoreCenterState();
  }, liftDelay);
  state.attentionMatrix.scoreTimers.push(moveTimer);

  const landingTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 10) return;
    slide.classList.remove('attn23-show-score-operands');
    slide.classList.remove('attn23-show-score-transposed');
    slide.classList.remove('attn23-show-score-formula');
    slide.classList.remove('attn23-show-score-matrix');
    slide.classList.remove('attn23-show-score-mask-focus');
    slide.classList.remove('attn23-show-score-future-problem');
    slide.classList.remove('attn23-show-score-causal-mask');
    slide.classList.remove('attn23-show-score-mask-question');
    slide.classList.add('attn23-show-postscore-centered');
    syncAttentionMatrixScoreStageFromClasses();
    ghost.style.opacity = '0';
  }, liftDelay + ATTN_MATRIX_POSTSCORE_CENTER_MS - 90);
  state.attentionMatrix.scoreTimers.push(landingTimer);

  const cleanupGhostTimer = setTimeout(() => {
    if (ghost.parentNode) ghost.parentNode.removeChild(ghost);
  }, liftDelay + ATTN_MATRIX_POSTSCORE_CENTER_MS + ATTN_MATRIX_FADE_MS + 40);
  state.attentionMatrix.scoreTimers.push(cleanupGhostTimer);

  const settleTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 10) return;
    settleAttentionMatrixPostScoreCenterState();
  }, liftDelay + ATTN_MATRIX_POSTSCORE_CENTER_MS + ATTN_MATRIX_FADE_MS + 60);
  state.attentionMatrix.scoreTimers.push(settleTimer);
}

function runAttentionMatrixScaleSequence() {
  const slide = document.getElementById('slide-23');
  if (!slide) return;

  settleAttentionMatrixPostScoreCenterState();
  clearAttentionMatrixScoreTimers();
  state.attentionMatrix.scaledMatrixDone = false;
  state.attentionMatrix.postScoreMode = 'scaled';
  syncAttentionMatrixPostScoreMatrix('masked_raw');
  slide.classList.remove('attn23-show-postscore-scale');
  syncAttentionMatrixScoreStageFromClasses();

  const scaleTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 11) return;
    slide.classList.add('attn23-show-postscore-scale');
    syncAttentionMatrixPostScoreMatrix('scaled');
    syncAttentionMatrixScoreStageFromClasses();
  }, 30);
  state.attentionMatrix.scoreTimers.push(scaleTimer);

  const settleTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 11) return;
    settleAttentionMatrixScaleState();
  }, 30 + ATTN_MATRIX_SCALE_FADE_MS);
  state.attentionMatrix.scoreTimers.push(settleTimer);
}

function runAttentionMatrixSoftmaxSequence() {
  const slide = document.getElementById('slide-23');
  const rowSum = document.getElementById('attn23-postscore-row-sum');
  if (!slide) return;

  settleAttentionMatrixScaleState();
  clearAttentionMatrixScoreTimers();
  clearAttentionMatrixPostScoreRowActivity();
  state.attentionMatrix.attentionMatrixDone = false;
  state.attentionMatrix.postScoreMode = 'attention';
  state.attentionMatrix.postScoreVisibleCount = 0;
  syncAttentionMatrixPostScoreRows(0);
  syncAttentionMatrixPostScoreMatrix('attention');
  slide.classList.remove('attn23-show-postscore-scale');
  slide.classList.add('attn23-show-postscore-softmax');
  if (rowSum) {
    rowSum.style.setProperty('opacity', '0', 'important');
    rowSum.style.setProperty('visibility', 'hidden', 'important');
  }
  syncAttentionMatrixScoreStageFromClasses();

  ATTN_MATRIX_SCORE_TOKENS.forEach((token, idx) => {
    const timer = setTimeout(() => {
      if (state.attentionMatrix.step < 12) return;
      clearAttentionMatrixPostScoreRowActivity();
      const row = document.getElementById('attn23-postscore-row-' + token);
      if (row) row.classList.add('is-softmax-active');
      state.attentionMatrix.postScoreVisibleCount = Math.max(state.attentionMatrix.postScoreVisibleCount, idx + 1);
      syncAttentionMatrixPostScoreRows(state.attentionMatrix.postScoreVisibleCount);
      syncAttentionMatrixPostScoreMatrix('attention');
    }, 40 + (idx * ATTN_MATRIX_SOFTMAX_ROW_STAGGER_MS));
    state.attentionMatrix.scoreTimers.push(timer);

    const clearTimer = setTimeout(() => {
      const row = document.getElementById('attn23-postscore-row-' + token);
      if (row) row.classList.remove('is-softmax-active');
    }, 40 + (idx * ATTN_MATRIX_SOFTMAX_ROW_STAGGER_MS) + ATTN_MATRIX_SOFTMAX_ROW_MS);
    state.attentionMatrix.scoreTimers.push(clearTimer);
  });

  const settleTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 12) return;
    settleAttentionMatrixSoftmaxState();
  }, 40 + ((ATTN_MATRIX_SCORE_TOKENS.length - 1) * ATTN_MATRIX_SOFTMAX_ROW_STAGGER_MS) + ATTN_MATRIX_SOFTMAX_ROW_MS + 80);
  state.attentionMatrix.scoreTimers.push(settleTimer);
}

function runAttentionMatrixValueEntrySequence() {
  const slide = document.getElementById('slide-23');
  const rowSum = document.getElementById('attn23-postscore-row-sum');
  if (!slide) return;

  settleAttentionMatrixSoftmaxState();
  clearAttentionMatrixScoreTimers();
  clearAttentionMatrixPostScoreRowActivity();
  state.attentionMatrix.valueMatrixVisibleDone = false;
  if (rowSum) {
    rowSum.style.setProperty('opacity', '0', 'important');
    rowSum.style.setProperty('visibility', 'hidden', 'important');
  }
  syncAttentionMatrixScoreStageFromClasses();

  const entryTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 13) return;
    slide.classList.add('attn23-show-postscore-av');
    syncAttentionMatrixScoreStageFromClasses();
  }, 30);
  state.attentionMatrix.scoreTimers.push(entryTimer);

  const settleTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 13) return;
    settleAttentionMatrixValueEntryState();
  }, 30 + ATTN_MATRIX_VALUE_ENTRY_MS);
  state.attentionMatrix.scoreTimers.push(settleTimer);
}

function runAttentionMatrixOutputSequence() {
  const slide = document.getElementById('slide-23');
  const vWrap = document.getElementById('attn23-postscore-v-wrap');
  if (!slide) return;

  settleAttentionMatrixValueEntryState();
  clearAttentionMatrixScoreTimers();
  clearAttentionMatrixPostScoreRowActivity();
  state.attentionMatrix.outputMatrixDone = false;
  state.attentionMatrix.outputVisibleCount = 0;
  syncAttentionMatrixOutputRows(0);
  slide.classList.add('attn23-show-postscore-output');
  syncAttentionMatrixScoreStageFromClasses();

  ATTN_MATRIX_TOKENS.forEach((token, idx) => {
    const timer = setTimeout(() => {
      if (state.attentionMatrix.step < 14) return;
      clearAttentionMatrixPostScoreRowActivity();
      const row = document.getElementById('attn23-postscore-row-' + token);
      if (row) row.classList.add('is-output-active');
      if (vWrap) vWrap.classList.add('is-source-emphasis');
      state.attentionMatrix.outputVisibleCount = Math.max(state.attentionMatrix.outputVisibleCount, idx + 1);
      syncAttentionMatrixOutputRows(state.attentionMatrix.outputVisibleCount);
    }, 40 + (idx * ATTN_MATRIX_OUTPUT_ROW_STAGGER_MS));
    state.attentionMatrix.scoreTimers.push(timer);
  });

  const settleTimer = setTimeout(() => {
    if (state.attentionMatrix.step < 14) return;
    settleAttentionMatrixOutputState();
  }, 40 + ((ATTN_MATRIX_TOKENS.length - 1) * ATTN_MATRIX_OUTPUT_ROW_STAGGER_MS) + ATTN_MATRIX_OUTPUT_ROW_MS + 60);
  state.attentionMatrix.scoreTimers.push(settleTimer);
}

function animateAttentionMatrixTokenRow(token, idx, delayMs, isLastToken = false) {
  const timerId = setTimeout(() => {
    if (state.attentionMatrix.step < 1) return;
    const sourceWrap = document.getElementById('attn23-source-chip-wrap-' + token);
    const targetWrap = document.getElementById('attn23-token-chip-wrap-' + token);
    const ghost = createAttentionMatrixTokenGhost(token);
    if (!ghost || !sourceWrap || !targetWrap || !animateAttentionMatrixGhost(ghost, sourceWrap, targetWrap, ATTN_MATRIX_TOKEN_TRAVEL_MS)) {
      state.attentionMatrix.tokenVisibleCount = Math.max(state.attentionMatrix.tokenVisibleCount, idx + 1);
      syncAttentionMatrixTokenSlots(state.attentionMatrix.tokenVisibleCount);
      if (isLastToken) settleAttentionMatrixTokenState();
      return;
    }

    const revealTimer = setTimeout(() => {
      if (state.attentionMatrix.step < 1) return;
      state.attentionMatrix.tokenVisibleCount = Math.max(state.attentionMatrix.tokenVisibleCount, idx + 1);
      syncAttentionMatrixTokenSlots(state.attentionMatrix.tokenVisibleCount);
      ghost.style.opacity = '0';
    }, ATTN_MATRIX_TOKEN_TRAVEL_MS);
    state.attentionMatrix.timers.push(revealTimer);

    const cleanupTimer = setTimeout(() => {
      if (ghost.parentNode) ghost.parentNode.removeChild(ghost);
      if (isLastToken) settleAttentionMatrixTokenState();
    }, ATTN_MATRIX_TOKEN_TRAVEL_MS + ATTN_MATRIX_FADE_MS + 40);
    state.attentionMatrix.timers.push(cleanupTimer);
  }, delayMs);

  state.attentionMatrix.timers.push(timerId);
}

function animateAttentionMatrixXRow(token, idx, delayMs, isLastToken = false) {
  const timerId = setTimeout(() => {
    if (state.attentionMatrix.step < 2) return;
    const sourceWrap = document.getElementById('attn23-source-vector-wrap-' + token);
    const targetWrap = document.getElementById('attn23-x-vector-wrap-' + token);
    const ghost = createAttentionMatrixXGhost(token);
    if (!ghost || !sourceWrap || !targetWrap || !animateAttentionMatrixGhost(ghost, sourceWrap, targetWrap, ATTN_MATRIX_X_TRAVEL_MS)) {
      state.attentionMatrix.xVisibleCount = Math.max(state.attentionMatrix.xVisibleCount, idx + 1);
      syncAttentionMatrixXSlots(state.attentionMatrix.xVisibleCount);
      if (isLastToken) settleAttentionMatrixFullState();
      return;
    }

    const revealTimer = setTimeout(() => {
      if (state.attentionMatrix.step < 2) return;
      state.attentionMatrix.xVisibleCount = Math.max(state.attentionMatrix.xVisibleCount, idx + 1);
      syncAttentionMatrixXSlots(state.attentionMatrix.xVisibleCount);
      ghost.style.opacity = '0';
    }, ATTN_MATRIX_X_TRAVEL_MS);
    state.attentionMatrix.timers.push(revealTimer);

    const cleanupTimer = setTimeout(() => {
      if (ghost.parentNode) ghost.parentNode.removeChild(ghost);
      if (isLastToken) settleAttentionMatrixFullState();
    }, ATTN_MATRIX_X_TRAVEL_MS + ATTN_MATRIX_FADE_MS + 40);
    state.attentionMatrix.timers.push(cleanupTimer);
  }, delayMs);

  state.attentionMatrix.timers.push(timerId);
}

function runAttentionMatrixTokenSequence() {
  const slide = document.getElementById('slide-23');
  resetAttentionMatrixVisuals();
  if (slide) slide.classList.add('attn23-show-token-matrix');

  if (!ATTN_MATRIX_TOKENS.length) {
    settleAttentionMatrixTokenState();
    return;
  }

  ATTN_MATRIX_TOKENS.forEach((token, idx) => {
    animateAttentionMatrixTokenRow(
      token,
      idx,
      idx * ATTN_MATRIX_ROW_STAGGER_MS,
      idx === ATTN_MATRIX_TOKENS.length - 1
    );
  });
}

function runAttentionMatrixXSequence() {
  const slide = document.getElementById('slide-23');
  if (!state.attentionMatrix.tokenMatrixDone) {
    settleAttentionMatrixTokenState();
  }

  clearAttentionMatrixTimers();
  clearAttentionMatrixGhosts();
  state.attentionMatrix.xMatrixDone = false;
  state.attentionMatrix.xVisibleCount = 0;
  syncAttentionMatrixXSlots(0);
  if (slide) {
    slide.classList.add('attn23-show-token-matrix');
    slide.classList.add('attn23-show-x-matrix');
    slide.classList.add('attn23-token-settled');
    slide.classList.remove('attn23-full-settled');
  }

  if (!ATTN_MATRIX_TOKENS.length) {
    settleAttentionMatrixFullState();
    return;
  }

  ATTN_MATRIX_TOKENS.forEach((token, idx) => {
    animateAttentionMatrixXRow(
      token,
      idx,
      idx * ATTN_MATRIX_ROW_STAGGER_MS,
      idx === ATTN_MATRIX_TOKENS.length - 1
    );
  });
}

function initAttentionMatrixSlide() {
  const slide = document.getElementById('slide-23');
  const sourceRow = document.getElementById('attn23-source-row');
  const tokenSlots = document.getElementById('attn23-token-slots');
  const xSlots = document.getElementById('attn23-x-slots');
  const projRow = document.getElementById('attn23-proj-row');
  const scoreStage = document.getElementById('attn23-score-stage');
  if (!slide || !sourceRow || !tokenSlots || !xSlots || !projRow || !scoreStage) return;

  if (!state.attentionMatrix.initialized) {
    sourceRow.innerHTML = '';
    tokenSlots.innerHTML = '';
    xSlots.innerHTML = '';
    projRow.innerHTML = '';
    scoreStage.innerHTML = '';

    ATTN_MATRIX_TOKENS.forEach((token) => {
      sourceRow.appendChild(createAttentionMatrixSourceItem(token));

      const tokenSlot = createEl('div', {
        className: 'attn23-slot attn23-token-slot',
        id: 'attn23-token-slot-' + token,
        dataset: { token }
      });
      tokenSlot.appendChild(createAttentionMatrixTokenItem(token));
      tokenSlots.appendChild(tokenSlot);

      const xSlot = createEl('div', {
        className: 'attn23-slot attn23-x-slot',
        id: 'attn23-x-slot-' + token,
        dataset: { token }
      });
      xSlot.appendChild(createAttentionMatrixXItem(token));
      xSlots.appendChild(xSlot);
    });
    ATTN_MATRIX_PROJS.forEach((proj) => {
      projRow.appendChild(createAttentionMatrixProjectionColumn(proj));
    });
    scoreStage.appendChild(createAttentionMatrixScoreStage());

    if (!state.attentionMatrix.resizeBound) {
      addTrackedListener(window, 'resize', () => {
        if (!state.attentionMatrix.initialized) return;
        if (state.attentionMatrix.step >= 14) {
          settleAttentionMatrixOutputState();
        } else if (state.attentionMatrix.step >= 13) {
          settleAttentionMatrixValueEntryState();
        } else if (state.attentionMatrix.step >= 12) {
          settleAttentionMatrixSoftmaxState();
        } else if (state.attentionMatrix.step >= 11) {
          settleAttentionMatrixScaleState();
        } else if (state.attentionMatrix.step >= 10) {
          settleAttentionMatrixPostScoreCenterState();
        } else if (state.attentionMatrix.step >= 9) {
          settleAttentionMatrixMaskQuestionState();
        } else if (state.attentionMatrix.step >= 8) {
          settleAttentionMatrixCausalMaskState();
        } else if (state.attentionMatrix.step >= 7) {
          settleAttentionMatrixMaskProblemState();
        } else if (state.attentionMatrix.step >= 6) {
          settleAttentionMatrixScoreMatrixState();
        } else if (state.attentionMatrix.step >= 5) {
          settleAttentionMatrixScoreTransposeState();
        } else if (state.attentionMatrix.step >= 4) {
          settleAttentionMatrixScoreCenterState();
        } else if (state.attentionMatrix.step >= 3) {
          settleAttentionMatrixProjectionState();
        } else if (state.attentionMatrix.step >= 2) {
          settleAttentionMatrixFullState();
        } else if (state.attentionMatrix.step >= 1) {
          settleAttentionMatrixTokenState();
        } else {
          resetAttentionMatrixVisuals();
        }
      });
      state.attentionMatrix.resizeBound = true;
    }

    state.attentionMatrix.initialized = true;
  }

  const takeaway = document.getElementById('attn23-takeaway');
  if (takeaway) setMathHTML(takeaway, ATTN_MATRIX_TAKEAWAYS[state.attentionMatrix.step] || ATTN_MATRIX_TAKEAWAYS[0]);
  typesetMath(slide);

  if (state.attentionMatrix.step >= 14) {
    settleAttentionMatrixOutputState();
  } else if (state.attentionMatrix.step >= 13) {
    settleAttentionMatrixValueEntryState();
  } else if (state.attentionMatrix.step >= 12) {
    settleAttentionMatrixSoftmaxState();
  } else if (state.attentionMatrix.step >= 11) {
    settleAttentionMatrixScaleState();
  } else if (state.attentionMatrix.step >= 10) {
    settleAttentionMatrixPostScoreCenterState();
  } else if (state.attentionMatrix.step >= 9) {
    settleAttentionMatrixMaskQuestionState();
  } else if (state.attentionMatrix.step >= 8) {
    settleAttentionMatrixCausalMaskState();
  } else if (state.attentionMatrix.step >= 7) {
    settleAttentionMatrixMaskProblemState();
  } else if (state.attentionMatrix.step >= 6) {
    settleAttentionMatrixScoreMatrixState();
  } else if (state.attentionMatrix.step >= 5) {
    settleAttentionMatrixScoreTransposeState();
  } else if (state.attentionMatrix.step >= 4) {
    settleAttentionMatrixScoreCenterState();
  } else if (state.attentionMatrix.step >= 3) {
    settleAttentionMatrixProjectionState();
  } else if (state.attentionMatrix.step >= 2) {
    settleAttentionMatrixFullState();
  } else if (state.attentionMatrix.step >= 1) {
    settleAttentionMatrixTokenState();
  } else {
    resetAttentionMatrixVisuals();
  }
}

function setAttentionMatrixStep(step) {
  const slide = document.getElementById('slide-23');
  const takeaway = document.getElementById('attn23-takeaway');
  if (!slide || !takeaway) return;

  const prevStep = state.attentionMatrix.step;
  const clamped = Math.max(0, Math.min(ATTN_MATRIX_MAX_STEP, step));
  state.attentionMatrix.step = clamped;
  setMathHTML(takeaway, ATTN_MATRIX_TAKEAWAYS[clamped] || ATTN_MATRIX_TAKEAWAYS[0]);
  const animateStep = clamped === prevStep + 1;

  if (clamped < 3) {
    resetAttentionMatrixProjectionVisuals();
  }
  if (clamped < 4) {
    resetAttentionMatrixScoreVisuals();
  }
  if (clamped < 7) {
    resetAttentionMatrixScoreMaskVisuals();
  }
  if (clamped < 10) {
    resetAttentionMatrixPostScoreVisuals();
  }

  if (clamped === 0) {
    resetAttentionMatrixVisuals();
    return;
  }

  if (clamped === 1) {
    if (!state.attentionMatrix.tokenMatrixDone && animateStep) {
      runAttentionMatrixTokenSequence();
    } else {
      settleAttentionMatrixTokenState();
    }
    return;
  }

  if (clamped === 2) {
    if (!state.attentionMatrix.tokenMatrixDone) {
      settleAttentionMatrixTokenState();
    }
    if (!state.attentionMatrix.xMatrixDone && animateStep) {
      runAttentionMatrixXSequence();
    } else {
      settleAttentionMatrixFullState();
    }
    return;
  }

  if (!state.attentionMatrix.tokenMatrixDone) {
    settleAttentionMatrixTokenState();
  }
  if (!state.attentionMatrix.xMatrixDone) {
    settleAttentionMatrixFullState();
  }
  if (clamped === 3) {
    if (!state.attentionMatrix.projectionDone && animateStep) {
      runAttentionMatrixProjectionSequence();
    } else {
      settleAttentionMatrixProjectionState();
    }
    return;
  }

  if (!state.attentionMatrix.projectionDone) {
    settleAttentionMatrixProjectionState();
  }
  if (clamped === 4) {
    if (!state.attentionMatrix.scoreCenteredDone && animateStep) {
      runAttentionMatrixScoreCenterSequence();
    } else {
      settleAttentionMatrixScoreCenterState();
    }
    return;
  }

  if (!state.attentionMatrix.scoreCenteredDone) {
    settleAttentionMatrixScoreCenterState();
  }
  if (clamped === 5) {
    if (!state.attentionMatrix.scoreTransposedDone && animateStep) {
      runAttentionMatrixScoreTransposeSequence();
    } else {
      settleAttentionMatrixScoreTransposeState();
    }
    return;
  }

  if (!state.attentionMatrix.scoreTransposedDone) {
    settleAttentionMatrixScoreTransposeState();
  }

  if (clamped === 6) {
    if (!state.attentionMatrix.scoreMatrixDone && animateStep) {
      runAttentionMatrixScoreMatrixSequence();
    } else {
      settleAttentionMatrixScoreMatrixState();
    }
    return;
  }

  if (!state.attentionMatrix.scoreMatrixDone) {
    settleAttentionMatrixScoreMatrixState();
  }

  if (clamped === 7) {
    if (!state.attentionMatrix.maskProblemDone && animateStep) {
      runAttentionMatrixMaskProblemSequence();
    } else {
      settleAttentionMatrixMaskProblemState();
    }
    return;
  }

  if (!state.attentionMatrix.maskProblemDone) {
    settleAttentionMatrixMaskProblemState();
  }
  if (clamped === 8) {
    if (!state.attentionMatrix.maskAppliedDone && animateStep) {
      runAttentionMatrixCausalMaskSequence();
    } else {
      settleAttentionMatrixCausalMaskState();
    }
    return;
  }

  if (!state.attentionMatrix.maskAppliedDone) {
    settleAttentionMatrixCausalMaskState();
  }
  if (clamped === 9) {
    if (!state.attentionMatrix.maskQuestionDone && animateStep) {
      runAttentionMatrixMaskQuestionSequence();
    } else {
      settleAttentionMatrixMaskQuestionState();
    }
    return;
  }

  if (!state.attentionMatrix.maskQuestionDone) {
    settleAttentionMatrixMaskQuestionState();
  }
  if (clamped === 10) {
    if (!state.attentionMatrix.postScoreCenteredDone && animateStep) {
      runAttentionMatrixPostScoreCenterSequence();
    } else {
      settleAttentionMatrixPostScoreCenterState();
    }
    return;
  }

  if (!state.attentionMatrix.postScoreCenteredDone) {
    settleAttentionMatrixPostScoreCenterState();
  }
  if (clamped === 11) {
    if (!state.attentionMatrix.scaledMatrixDone && animateStep) {
      runAttentionMatrixScaleSequence();
    } else {
      settleAttentionMatrixScaleState();
    }
    return;
  }

  if (!state.attentionMatrix.scaledMatrixDone) {
    settleAttentionMatrixScaleState();
  }
  if (clamped === 12) {
    if (!state.attentionMatrix.attentionMatrixDone && animateStep) {
      runAttentionMatrixSoftmaxSequence();
    } else {
      settleAttentionMatrixSoftmaxState();
    }
    return;
  }

  if (!state.attentionMatrix.attentionMatrixDone) {
    settleAttentionMatrixSoftmaxState();
  }
  if (clamped === 13) {
    if (!state.attentionMatrix.valueMatrixVisibleDone && animateStep) {
      runAttentionMatrixValueEntrySequence();
    } else {
      settleAttentionMatrixValueEntryState();
    }
    return;
  }

  if (!state.attentionMatrix.valueMatrixVisibleDone) {
    settleAttentionMatrixValueEntryState();
  }
  if (clamped === 14) {
    if (!state.attentionMatrix.outputMatrixDone && animateStep) {
      runAttentionMatrixOutputSequence();
    } else {
      settleAttentionMatrixOutputState();
    }
  }
}

function runAttentionMatrixStep() {
  if (!state.attentionMatrix.initialized) initAttentionMatrixSlide();
  if (state.attentionMatrix.step >= ATTN_MATRIX_MAX_STEP) return false;
  setAttentionMatrixStep(state.attentionMatrix.step + 1);
  return true;
}

function resetAttentionMatrixSlide() {
  const slide = document.getElementById('slide-23');
  if (!slide) return;
  setAttentionMatrixStep(0);
}
