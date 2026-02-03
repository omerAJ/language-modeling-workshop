"""
Language Modeling Forward Pass Animation (Simplified)
=====================================================
A reusable Manim animation showing the forward pass of a transformer
language model up to positional embeddings.

Render commands:
    manim -pqh pipeline.py LanguageModelingPipeline
    manim -pql pipeline.py LanguageModelingPipeline  # fast preview

Manim Community Edition v0.18+
"""

from manim import *
import numpy as np

# =============================================================================
# PARAMETERS
# =============================================================================

SENTENCE = "The cat sat on the"
TOKENS = ["The", "cat", "sat", "on", "the"]
TOKEN_IDS = [464, 3797, 3332, 319, 262]
T = 5  # sequence length
D_MODEL = 8  # visual cell count for the d dimension (labels show "d")

# Autoregressive iterations for the black-box demonstration
BLACKBOX_ITERATIONS = [
    {
        "selected": "mat",
        "top_tokens": ["mat", "floor", "couch", "bed", "..."],
        "probs": [0.42, 0.18, 0.15, 0.12, 0.13],
    },
    {
        "selected": ".",
        "top_tokens": [".", ",", "and", "!", "..."],
        "probs": [0.55, 0.20, 0.10, 0.08, 0.07],
    },
]

# Visual constants
DARK_BG = "#1a1a2e"
ACCENT_COLOR = "#4fc3f7"
ACCENT_COLOR_2 = "#81c784"
TEXT_COLOR = WHITE
BOX_COLOR = "#2d2d44"
LLM_BOX_FILL = "#0d0d1a"
MATRIX_COLORS = [BLUE_E, BLUE_D, BLUE_C, BLUE_B, BLUE_A, GREEN_A, YELLOW_A, ORANGE]


# =============================================================================
# REUSABLE CLASSES
# =============================================================================

class LLMBlackBox(VGroup):
    """Opaque 'black box' representation of an LLM."""

    def __init__(self, width=2.8, height=1.8, **kwargs):
        super().__init__(**kwargs)

        self.box = RoundedRectangle(
            width=width, height=height, corner_radius=0.2,
            stroke_color=ACCENT_COLOR, stroke_width=3,
            fill_color=LLM_BOX_FILL, fill_opacity=0.95,
        )
        self.title_text = Text("LLM", font_size=36, color=ACCENT_COLOR, weight=BOLD)
        self.title_text.move_to(self.box.get_center() + UP * 0.15)

        self.add(self.box, self.title_text)


class CompactDistribution(VGroup):
    """Compact horizontal-bar probability distribution."""

    def __init__(self, tokens, probs, bar_max_width=1.4, **kwargs):
        super().__init__(**kwargs)
        self.token_groups = VGroup()
        max_prob = max(probs)

        rows = []
        for i, (tok, prob) in enumerate(zip(tokens, probs)):
            label = Text(tok, font_size=16, color=TEXT_COLOR)
            bar = Rectangle(
                width=max((prob / max_prob) * bar_max_width, 0.06),
                height=0.22,
                fill_color=ACCENT_COLOR, fill_opacity=0.7,
                stroke_width=0,
            )
            pct = Text(f"{prob:.0%}", font_size=12, color=GRAY_B)
            label.shift(DOWN * i * 0.38)
            rows.append((label, bar, pct))

        max_label_right = max(r[0].get_right()[0] for r in rows)
        bar_x = max_label_right + 0.25

        for label, bar, pct in rows:
            bar.move_to(np.array([bar_x + bar.width / 2, label.get_center()[1], 0]))
            pct.next_to(bar, RIGHT, buff=0.1)
            self.token_groups.add(VGroup(label, bar, pct))

        self.add(self.token_groups)
        self.center()


class TokenRow(VGroup):
    """Row of token boxes."""

    def __init__(self, tokens, box_width=1.2, box_height=0.6, spacing=0.15, **kwargs):
        super().__init__(**kwargs)
        self.tokens = tokens
        self.boxes = VGroup()
        self.labels = VGroup()

        for i, token in enumerate(tokens):
            box = RoundedRectangle(
                width=box_width, height=box_height,
                corner_radius=0.1, stroke_color=ACCENT_COLOR, stroke_width=2,
                fill_color=BOX_COLOR, fill_opacity=0.8,
            )
            label = Text(f'"{token}"', font_size=20, color=TEXT_COLOR)
            label.move_to(box.get_center())
            token_group = VGroup(box, label)
            token_group.shift(RIGHT * i * (box_width + spacing))
            self.boxes.add(box)
            self.labels.add(label)
            self.add(token_group)
        self.center()


class IdRow(VGroup):
    """Row of integer token-ID boxes."""

    def __init__(self, ids, box_width=1.2, box_height=0.5, spacing=0.15, **kwargs):
        super().__init__(**kwargs)
        self.ids = ids
        self.boxes = VGroup()
        self.labels = VGroup()

        for i, token_id in enumerate(ids):
            box = RoundedRectangle(
                width=box_width, height=box_height,
                corner_radius=0.1, stroke_color=ACCENT_COLOR_2, stroke_width=2,
                fill_color=BOX_COLOR, fill_opacity=0.8,
            )
            label = Text(str(token_id), font_size=18, color=ACCENT_COLOR_2)
            label.move_to(box.get_center())
            id_group = VGroup(box, label)
            id_group.shift(RIGHT * i * (box_width + spacing))
            self.boxes.add(box)
            self.labels.add(label)
            self.add(id_group)
        self.center()


class TensorMatrix(VGroup):
    """Abstract heatmap grid representing a tensor."""

    def __init__(self, rows, cols, cell_size=0.25, label=None, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows
        self.cols = cols
        self.cells = VGroup()

        np.random.seed(42)
        for i in range(rows):
            for j in range(cols):
                idx = min(
                    int((i + j) / (rows + cols) * len(MATRIX_COLORS)),
                    len(MATRIX_COLORS) - 1,
                )
                fill_color = MATRIX_COLORS[idx]
                opacity = 0.3 + 0.5 * np.random.random()
                cell = Rectangle(
                    width=cell_size, height=cell_size,
                    stroke_color=WHITE, stroke_width=0.5,
                    fill_color=fill_color, fill_opacity=opacity,
                )
                cell.move_to(RIGHT * j * cell_size + DOWN * i * cell_size)
                self.cells.add(cell)

        self.add(self.cells)
        self.cells.center() 

        self.border = Rectangle(
            width=cols * cell_size + 0.05,
            height=rows * cell_size + 0.05,
            stroke_color=WHITE, stroke_width=2, fill_opacity=0,
        )
        self.border.move_to(self.cells.get_center())
        self.add(self.border)

        if label:
            self.label = Text(label, font_size=16, color=GRAY_B)
            self.label.next_to(self, DOWN, buff=0.15)
            self.add(self.label)

    def get_cell(self, row, col):
        return self.cells[row * self.cols + col]


class EmbeddingWeightMatrix(VGroup):
    """Embedding weight matrix W_E ∈ ℝ^{|V|×d} with ellipsis to show scale."""

    def __init__(self, num_cols=8, cell_size=0.22, top_rows=3, bottom_rows=2, **kwargs):
        super().__init__(**kwargs)
        self.num_cols = num_cols
        self.cell_size = cell_size

        np.random.seed(99)

        # Top visible rows
        self.top_section = VGroup(*[self._make_row(i) for i in range(top_rows)])
        self.top_section.arrange(DOWN, buff=0)

        # Ellipsis dots spread across the row width
        ellipsis_dots = VGroup()
        for _ in range(min(num_cols, 4)):
            d = MathTex(r"\vdots", font_size=12, color=GRAY_B)
            ellipsis_dots.add(d)
        ellipsis_dots.arrange(RIGHT, buff=0.3)
        self.ellipsis = ellipsis_dots

        # Bottom visible rows
        self.bottom_section = VGroup(*[self._make_row(50 + i) for i in range(bottom_rows)])
        self.bottom_section.arrange(DOWN, buff=0)

        # Assemble vertically
        content = VGroup(self.top_section, ellipsis_dots, self.bottom_section)
        content.arrange(DOWN, buff=0.1)
        self.content = content
        self.add(content)

        # Border
        self.border = SurroundingRectangle(
            content, buff=0.04,
            stroke_color=WHITE, stroke_width=2, fill_opacity=0,
        )
        self.add(self.border)

    def _make_row(self, seed):
        row = VGroup()
        for j in range(self.num_cols):
            idx = (seed * 3 + j * 7) % len(MATRIX_COLORS)
            fill_color = MATRIX_COLORS[idx]
            opacity = 0.3 + 0.5 * np.random.random()
            cell = Rectangle(
                width=self.cell_size, height=self.cell_size,
                stroke_color=WHITE, stroke_width=0.5,
                fill_color=fill_color, fill_opacity=opacity,
            )
            row.add(cell)
        row.arrange(RIGHT, buff=0)
        return row

    def make_extracted_row(self, seed):
        """Create a new row of cells representing an extracted embedding vector."""
        rng = np.random.RandomState(seed + 200)
        row = VGroup()
        for j in range(self.num_cols):
            idx = (seed * 5 + j * 3 + 2) % len(MATRIX_COLORS)
            fill_color = MATRIX_COLORS[idx]
            opacity = 0.35 + 0.45 * rng.random()
            cell = Rectangle(
                width=self.cell_size, height=self.cell_size,
                stroke_color=WHITE, stroke_width=0.5,
                fill_color=fill_color, fill_opacity=opacity,
            )
            row.add(cell)
        row.arrange(RIGHT, buff=0)
        return row


class EquationLabel(VGroup):
    """LaTeX equation with optional subtitle."""

    def __init__(self, equation_tex, description=None, font_size=35, **kwargs):
        super().__init__(**kwargs)
        self.equation = MathTex(equation_tex, font_size=font_size, color=TEXT_COLOR)
        self.add(self.equation)
        if description:
            self.description = Text(description, font_size=20, color=GRAY_B)
            self.description.next_to(self.equation, DOWN, buff=0.15)
            self.add(self.description)


# =============================================================================
# MAIN SCENE
# =============================================================================

class LanguageModelingPipeline(Scene):
    """Black-box demo → zoom-in → tokenization → embeddings → positional embeddings."""

    def construct(self):
        self.camera.background_color = DARK_BG

        self.show_black_box_overview()
        self.show_zoom_transition()
        self.show_tokenization()
        self.show_token_ids()
        self.show_embeddings()
        self.show_positional_embeddings()
        self.wait(2)

    # ------------------------------------------------------------------
    # PHASE 1 — Black-box overview
    # ------------------------------------------------------------------

    def show_black_box_overview(self):
        """Title, LLM black box, and two autoregressive iterations."""

        title = Text(
            "How Language Models Generate Text",
            font_size=30, color=ACCENT_COLOR,
        )
        title.to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=1)

        llm_box = LLMBlackBox()
        llm_box.move_to(ORIGIN)
        self.play(FadeIn(llm_box, scale=0.85), run_time=1)

        current_text = SENTENCE
        sentence = Text(f'"{current_text}"', font_size=20, color=TEXT_COLOR)
        sentence.next_to(llm_box, LEFT, buff=1.6)

        arrow_in = Arrow(
            sentence.get_right(), llm_box.get_left(),
            buff=0.2, color=GRAY_B, stroke_width=2,
        )
        self.play(FadeIn(sentence, shift=RIGHT * 0.3), GrowArrow(arrow_in), run_time=1)
        self.wait(0.5)

        self.bb_title = title
        self.bb_box = llm_box
        self.bb_sentence = sentence
        self.bb_arrow_in = arrow_in

        for iteration in BLACKBOX_ITERATIONS:
            current_text = self._run_iteration(iteration, current_text)

    def _run_iteration(self, data, current_text):
        """One autoregressive step: process → distribution → select → append."""
        llm = self.bb_box

        self.play(
            llm.box.animate.set_stroke(YELLOW, width=5),
            run_time=0.25,
        )
        self.play(
            llm.box.animate.set_stroke(ACCENT_COLOR, width=3),
            run_time=0.25,
        )

        arrow_out = Arrow(
            llm.get_right(), llm.get_right() + RIGHT * 0.9,
            buff=0.15, color=GRAY_B, stroke_width=2,
        )
        self.play(GrowArrow(arrow_out), run_time=0.4)

        dist = CompactDistribution(data["top_tokens"], data["probs"])
        dist.next_to(arrow_out, RIGHT, buff=0.25)
        self.play(
            LaggedStart(
                *[FadeIn(r, shift=LEFT * 0.2) for r in dist.token_groups],
                lag_ratio=0.08,
            ),
            run_time=0.8,
        )

        sel_idx = data["top_tokens"].index(data["selected"])
        highlight = SurroundingRectangle(
            dist.token_groups[sel_idx],
            color=YELLOW, stroke_width=2, buff=0.06,
        )
        self.play(Create(highlight), run_time=0.4)
        self.wait(0.4)

        selected = data["selected"]
        flying = Text(selected, font_size=22, color=YELLOW, weight=BOLD)
        flying.move_to(dist.token_groups[sel_idx][0])

        target = self.bb_sentence.get_right() + RIGHT * 0.25 + UP * 0.25
        self.play(FadeIn(flying, scale=1.2), run_time=0.2)
        self.play(
            flying.animate(path_arc=-PI / 3).move_to(target),
            run_time=0.9,
            rate_func=smooth,
        )

        if selected in {".", ",", "!", "?", ";", ":"}:
            new_text = current_text + selected
        else:
            new_text = current_text + " " + selected

        new_sentence = Text(f'"{new_text}"', font_size=20, color=TEXT_COLOR)
        new_sentence.move_to(self.bb_sentence.get_center())

        new_arrow_in = Arrow(
            new_sentence.get_right(), self.bb_box.get_left(),
            buff=0.2, color=GRAY_B, stroke_width=2,
        )

        self.play(
            FadeOut(flying),
            FadeOut(dist), FadeOut(highlight), FadeOut(arrow_out),
            Transform(self.bb_sentence, new_sentence),
            Transform(self.bb_arrow_in, new_arrow_in),
            run_time=0.7,
        )
        self.wait(0.3)
        return new_text

    # ------------------------------------------------------------------
    # PHASE 2 — Zoom transition
    # ------------------------------------------------------------------

    def show_zoom_transition(self):
        """Animate 'entering' the LLM black box to reveal internals."""

        question = Text("But how does this work?", font_size=28, color=ACCENT_COLOR)
        question.next_to(self.bb_box, DOWN, buff=0.9)
        self.play(Write(question), run_time=1.2)
        self.wait(0.8)

        self.play(
            FadeOut(self.bb_title),
            FadeOut(self.bb_sentence),
            FadeOut(self.bb_arrow_in),
            FadeOut(question),
            run_time=0.8,
        )

        self.play(self.bb_box.animate.move_to(ORIGIN), run_time=0.4)

        self.play(
            self.bb_box.box.animate.set_stroke(YELLOW, width=6),
            rate_func=there_and_back,
            run_time=0.8,
        )

        self.play(
            self.bb_box.title_text.animate.set_opacity(0),
            run_time=0.5,
        )

        zoom_cover = self.bb_box.box.copy()
        zoom_cover.set_fill(DARK_BG, opacity=1)
        zoom_cover.set_stroke(ACCENT_COLOR, width=3)
        self.add(zoom_cover)
        self.remove(self.bb_box)

        self.play(
            zoom_cover.animate.scale(6).set_stroke(width=0),
            run_time=2.0,
            rate_func=rush_into,
        )

        inner_title = Text(
            "Inside the Language Model",
            font_size=32, color=ACCENT_COLOR,
        )
        inner_title.to_edge(UP, buff=0.5)

        sentence_label = Text("Input:", font_size=20, color=GRAY_B)
        sentence_text = Text(f'"{SENTENCE}"', font_size=28, color=TEXT_COLOR)
        sentence_group = VGroup(sentence_label, sentence_text).arrange(RIGHT, buff=0.3)
        sentence_group.next_to(inner_title, DOWN, buff=0.8)

        self.add_behind(inner_title, sentence_label, sentence_text)

        self.play(
            FadeOut(zoom_cover),
            run_time=1.5,
            rate_func=rush_from,
        )

        self.title = inner_title
        self.sentence_text = sentence_text
        self.sentence_label = sentence_label

    def add_behind(self, *mobjects):
        for m in mobjects:
            self.add(m)
            self.bring_to_back(m)

    # ------------------------------------------------------------------
    # PHASE 3 — Detailed pipeline (up to positional embeddings)
    # ------------------------------------------------------------------

    def show_tokenization(self):
        """Text → token boxes."""
        stage_label = Text("1. Tokenization", font_size=24, color=ACCENT_COLOR_2)
        stage_label.to_edge(LEFT, buff=0.5).shift(UP * 2)

        token_row = TokenRow(TOKENS)
        token_row.next_to(self.sentence_text, DOWN, buff=1)

        arrow = Arrow(
            self.sentence_text.get_bottom(), token_row.get_top(),
            buff=0.2, color=GRAY_B, stroke_width=2,
        )

        eq = EquationLabel(
            r"\text{text} \rightarrow [\text{tok}_1, \ldots, \text{tok}_T]"
        )
        eq.scale(0.7).to_edge(RIGHT, buff=0.8).shift(UP * 1.5)

        self.play(FadeIn(stage_label))
        self.play(GrowArrow(arrow), run_time=0.5)
        self.play(
            LaggedStart(*[FadeIn(b, scale=0.8) for b in token_row], lag_ratio=0.1),
            run_time=1.5,
        )
        self.play(Write(eq), run_time=0.8)
        self.wait(0.5)

        self.token_row = token_row
        self.token_arrow = arrow
        self.stage_label = stage_label
        self.current_eq = eq

    def show_token_ids(self):
        """Tokens → integer IDs."""
        new_stage = Text("2. Token IDs", font_size=24, color=ACCENT_COLOR_2)
        new_stage.move_to(self.stage_label)

        id_row = IdRow(TOKEN_IDS)
        id_row.next_to(self.token_row, DOWN, buff=0.8)

        arrows = VGroup(*[
            Arrow(
                self.token_row.boxes[i].get_bottom(), id_row.boxes[i].get_top(),
                buff=0.1, color=GRAY_B, stroke_width=1.5,
                max_tip_length_to_length_ratio=0.2,
            )
            for i in range(len(TOKENS))
        ])

        new_eq = EquationLabel(r"\text{tok}_i \rightarrow \text{id}_i \in \mathbb{Z}")
        new_eq.scale(0.7).move_to(self.current_eq)

        self.play(Transform(self.stage_label, new_stage))
        self.play(
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.05),
            run_time=0.8,
        )
        self.play(
            LaggedStart(*[FadeIn(b, scale=0.8) for b in id_row], lag_ratio=0.1),
            Transform(self.current_eq, new_eq),
            run_time=1.2,
        )
        self.wait(0.3)

        self.id_row = id_row
        self.id_arrows = arrows

    def show_embeddings(self):
        """IDs → embedding-matrix lookup: show W_E, extract rows, build E."""
        # ── Rearrange previous elements ──
        self.play(
            FadeOut(self.title), FadeOut(self.sentence_text),
            FadeOut(self.sentence_label), FadeOut(self.token_arrow),
            self.token_row.animate.scale(0.65).to_edge(UP, buff=0.35),
            self.id_row.animate.scale(0.65).to_edge(UP, buff=1.05),
            FadeOut(self.id_arrows),
            run_time=1,
        )

        new_stage = Text("3. Embedding Lookup", font_size=24, color=ACCENT_COLOR_2)
        new_stage.move_to(self.stage_label).shift(RIGHT * 0.5)

        new_eq = EquationLabel(
            r"E = W_E[\,\text{ids}\,] \in \mathbb{R}^{t \times d}"
        )
        new_eq.scale(0.7).move_to(self.current_eq)

        # ── Weight matrix W_E: |V| × d ──
        weight_mat = EmbeddingWeightMatrix(num_cols=D_MODEL, cell_size=0.22)
        weight_mat.move_to(LEFT * 2.5 + DOWN * 0.5)

        w_label = MathTex(r"W_E", font_size=26, color=ACCENT_COLOR)
        w_label.next_to(weight_mat, DOWN, buff=0.2)

        brace_v = Brace(weight_mat, LEFT, buff=0.1, color=GRAY_B)
        v_label = MathTex(r"|V|", font_size=18, color=GRAY_B)
        v_label.next_to(brace_v, LEFT, buff=0.08)

        brace_d_w = Brace(weight_mat, UP, buff=0.1, color=GRAY_B)
        d_label_w = MathTex(r"d", font_size=18, color=GRAY_B)
        d_label_w.next_to(brace_d_w, UP, buff=0.05)

        weight_group = VGroup(
            weight_mat, w_label, brace_v, v_label, brace_d_w, d_label_w,
        )

        self.play(Transform(self.stage_label, new_stage))
        self.play(
            FadeIn(weight_mat, scale=0.9),
            FadeIn(w_label),
            FadeIn(brace_v), FadeIn(v_label),
            FadeIn(brace_d_w), FadeIn(d_label_w),
            Transform(self.current_eq, new_eq),
            run_time=1.5,
        )
        self.wait(0.5)

        # ── Row extraction: each token-ID → one row from W_E ──
        extracted_rows = VGroup()
        result_center = RIGHT * 2.5 + DOWN * 0.5
        first_row_y = result_center[1] + (T - 1) * weight_mat.cell_size / 2

        for i in range(T):
            row = weight_mat.make_extracted_row(TOKEN_IDS[i])
            row.move_to(weight_mat.get_center())

            target_y = first_row_y - i * weight_mat.cell_size
            target_pos = np.array([result_center[0], target_y, 0])

            # Highlight the current token ID
            id_hl = SurroundingRectangle(
                self.id_row[i], buff=0.04, color=YELLOW, stroke_width=2,
            )
            # Arrow from token ID to weight matrix
            top_center = weight_mat.border.get_top()
            offset = (i - (T - 1) / 2) * 0.15
            arrow = Arrow(
                self.id_row[i].get_bottom(),
                top_center + RIGHT * offset,
                buff=0.12, color=YELLOW, stroke_width=1.5,
                max_tip_length_to_length_ratio=0.12,
            )

            self.play(Create(id_hl), GrowArrow(arrow), run_time=0.25)
            self.play(FadeIn(row, scale=0.6), run_time=0.15)
            self.play(
                row.animate.move_to(target_pos),
                FadeOut(id_hl), FadeOut(arrow),
                run_time=0.4,
            )
            extracted_rows.add(row)

        # ── Border and labels around the stacked result: E (t × d) ──
        result_border = SurroundingRectangle(
            extracted_rows, buff=0.04,
            stroke_color=WHITE, stroke_width=2, fill_opacity=0,
        )
        e_label = MathTex(r"E", font_size=24, color=ACCENT_COLOR)
        e_label.next_to(result_border, DOWN, buff=0.15)

        brace_t = Brace(result_border, LEFT, buff=0.1, color=GRAY_B)
        t_label = MathTex(r"t", font_size=18, color=GRAY_B)
        t_label.next_to(brace_t, LEFT, buff=0.05)

        brace_d_e = Brace(result_border, UP, buff=0.1, color=GRAY_B)
        d_label_e = MathTex(r"d", font_size=18, color=GRAY_B)
        d_label_e.next_to(brace_d_e, UP, buff=0.05)

        result_decor = VGroup(
            result_border, e_label, brace_t, t_label, brace_d_e, d_label_e,
        )

        self.play(
            Create(result_border), FadeIn(e_label),
            FadeIn(brace_t), FadeIn(t_label),
            FadeIn(brace_d_e), FadeIn(d_label_e),
            run_time=0.8,
        )
        self.wait(0.5)

        # ── Transition: replace with a clean TensorMatrix for next phase ──
        emb_matrix = TensorMatrix(T, D_MODEL, cell_size=0.22, label="E: t × d")
        emb_matrix.move_to(LEFT * 3.5 + DOWN * 0.5)

        self.play(
            FadeOut(weight_group),
            FadeOut(result_decor),
            FadeOut(extracted_rows),
            FadeIn(emb_matrix),
            run_time=1,
        )
        self.wait(0.3)

        self.emb_matrix = emb_matrix

    