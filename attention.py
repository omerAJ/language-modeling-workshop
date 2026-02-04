"""
Multi-Head Attention Deep Dive Animation
=========================================
A Manim animation that starts from the transformer pipeline overview,
zooms into the MHA block, and animates the full Multi-Head Attention
process step-by-step.

Render commands:
    manim -pqh attention.py MultiHeadAttentionScene
    manim -pql attention.py MultiHeadAttentionScene  # fast preview

Manim Community Edition v0.18+
"""

from manim import *
from manim import CubicBezier
import numpy as np

from pipeline import (
    T, D_MODEL,
    DARK_BG, ACCENT_COLOR, ACCENT_COLOR_2, TEXT_COLOR,
    BOX_COLOR, LLM_BOX_FILL, MATRIX_COLORS, EQUATION_BG,
    EQUATION_STROKE, SLOW_FACTOR,
    TensorMatrix, EquationLabel,
)

# =============================================================================
# CONSTANTS
# =============================================================================

N_HEADS = 2
D_K = D_MODEL // N_HEADS  # 4

ATTN_COLOR = "#e57373"
HEAD_COLORS = ["#42a5f5", "#ab47bc"]
MASK_COLOR = "#ff1744"
SOFTMAX_COLOR = "#fdd835"
SCORE_COLOR = "#80cbc4"

# Pipeline block layout (mirrors pipeline.py)
BW, BH = 2.5, 0.38
GAP = 0.34
PIPELINE_X = -1.5

# Pre-computed attention data
np.random.seed(42)
RAW_SCORES = np.round(np.random.randn(T, T) * 1.5, 2)
SCALED_SCORES = np.round(RAW_SCORES / np.sqrt(D_K), 2)
CAUSAL_MASK = np.triu(np.full((T, T), -np.inf), k=1)
MASKED_SCORES = SCALED_SCORES + CAUSAL_MASK


def _row_softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


ATTN_WEIGHTS = np.round(_row_softmax(MASKED_SCORES), 2)


# =============================================================================
# REUSABLE CLASSES
# =============================================================================

class AttentionMatrix(VGroup):
    """t x t matrix with value-driven coloring and optional numeric overlays."""

    def __init__(
        self,
        values,
        cell_size=0.38,
        show_values=True,
        low_color=BLUE_E,
        high_color=YELLOW,
        mask_color=MASK_COLOR,
        label=None,
        value_font_size=10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.values = values
        self.cell_size = cell_size
        rows, cols = values.shape
        self.n_rows = rows
        self.n_cols = cols
        self.cells = VGroup()
        self.val_texts = VGroup()

        finite_vals = values[np.isfinite(values)]
        vmin = finite_vals.min() if len(finite_vals) else 0
        vmax = finite_vals.max() if len(finite_vals) else 1
        rng = vmax - vmin if vmax != vmin else 1.0

        for i in range(rows):
            for j in range(cols):
                v = values[i, j]
                if not np.isfinite(v):
                    fill = mask_color
                    opacity = 0.85
                else:
                    t_val = (v - vmin) / rng
                    fill = interpolate_color(
                        ManimColor(low_color), ManimColor(high_color), t_val
                    )
                    opacity = 0.4 + 0.5 * t_val

                cell = Rectangle(
                    width=cell_size, height=cell_size,
                    stroke_color=WHITE, stroke_width=0.5,
                    fill_color=fill, fill_opacity=opacity,
                )
                cell.move_to(RIGHT * j * cell_size + DOWN * i * cell_size)
                self.cells.add(cell)

                if show_values:
                    if not np.isfinite(v):
                        txt = MathTex(r"-\infty", font_size=value_font_size, color=WHITE)
                    else:
                        txt = Text(f"{v:.2f}", font_size=value_font_size, color=WHITE)
                    txt.move_to(cell.get_center())
                    self.val_texts.add(txt)

        self.add(self.cells)
        self.cells.center()
        if show_values:
            # Re-center texts to match cells
            for idx, txt in enumerate(self.val_texts):
                txt.move_to(self.cells[idx].get_center())
            self.add(self.val_texts)

        self.border = Rectangle(
            width=cols * cell_size + 0.05,
            height=rows * cell_size + 0.05,
            stroke_color=WHITE, stroke_width=2, fill_opacity=0,
        )
        self.border.move_to(self.cells.get_center())
        self.add(self.border)

        if label:
            self.label_text = Text(label, font_size=16, color=GRAY_B)
            self.label_text.next_to(self, DOWN, buff=0.15)
            self.add(self.label_text)


# =============================================================================
# MAIN SCENE
# =============================================================================

class MultiHeadAttentionScene(Scene):
    """Transformer pipeline → zoom into MHA → full attention walkthrough."""

    def play(self, *args, **kwargs):
        run_time = kwargs.pop("run_time", 1.0)
        kwargs["run_time"] = run_time * SLOW_FACTOR
        return super().play(*args, **kwargs)

    def wait(self, duration=1.0):
        return super().wait(duration * SLOW_FACTOR)

    def construct(self):
        self.camera.background_color = DARK_BG

        self.phase_1_pipeline_overview()
        self.phase_2_zoom_into_mha()
        self.phase_3_input_x()
        self.phase_4_qkv_projections()
        self.phase_5_head_splitting()
        self.phase_6_scaled_dot_product()
        self.phase_7_concatenate_heads()
        self.phase_8_output_projection()
        self.wait(2)

    # ------------------------------------------------------------------
    # PHASE 1 — Pipeline overview (recreated from pipeline.py)
    # ------------------------------------------------------------------

    def phase_1_pipeline_overview(self):
        title = Text(
            "Multi-Head Attention Deep Dive",
            font_size=30, color=ACCENT_COLOR,
        )
        title.to_edge(UP, buff=0.25)

        block_data = [
            ("Token Embeddings",                         ACCENT_COLOR, 13),
            ("+ Positional Embeddings",                  ACCENT_COLOR, 13),
            ("Masked Multi-Head Self-Attention\n\t\t\t\t\t(MHA)", "#e57373", 10),
            ("Add & Norm",                               "#ffb74d",   13),
            ("Feed Forward Network\n\t\t\t\t(FFN)",      "#e57373",   13),
            ("Add & Norm",                               "#ffb74d",   13),
            ("Linear Projection",                        ACCENT_COLOR, 13),
            ("Softmax",                                  ACCENT_COLOR, 13),
            ("Next-token Probabilities",                 YELLOW,      13),
        ]
        self.orig_colors = [c for _, c, _ in block_data]

        def make_block(label, color, fs):
            box = RoundedRectangle(
                width=BW, height=BH, corner_radius=0.08,
                stroke_color=color, stroke_width=2,
                fill_color=BOX_COLOR, fill_opacity=0.75,
            )
            txt = Text(label, font_size=fs, color=color)
            txt.move_to(box)
            return VGroup(box, txt)

        blocks = [make_block(*d) for d in block_data]
        pipeline = VGroup(*blocks)
        pipeline.arrange(UP, buff=GAP)
        pipeline.move_to(np.array([PIPELINE_X, -0.15, 0]))

        arrows = VGroup()
        for i in range(len(blocks) - 1):
            a = Arrow(
                blocks[i].get_top(), blocks[i + 1].get_bottom(),
                buff=0.02, color=GRAY_B, stroke_width=1.5,
                max_tip_length_to_length_ratio=0.25,
            )
            arrows.add(a)

        # Transformer block container
        tf_inner = VGroup(blocks[2], blocks[3], blocks[4], blocks[5])
        container = SurroundingRectangle(
            tf_inner, buff=0.25,
            stroke_color=GRAY_A, stroke_width=1.5, fill_opacity=0.05,
        )
        brace = Brace(container, RIGHT, buff=0.15, color=GRAY_B)
        n_label = MathTex(r"\times N", font_size=18, color=GRAY_B)
        n_label.next_to(brace, RIGHT, buff=0.08)

        # Residual arrows
        res_color = "#66bb6a"

        def residual_ortho(flow_arrow, dst_block, x_offset=1.6,
                           bend_radius=0.18, into_box=0.06, stroke_w=2.2):
            start = flow_arrow.get_center() + RIGHT * 0.05
            end_y = dst_block.get_center()[1]
            end_x = dst_block[0].get_right()[0] - into_box
            end = np.array([end_x, end_y, 0])
            bend_x = start[0] + x_offset
            A = start
            B = np.array([bend_x, start[1], 0])
            C = np.array([bend_x, end_y, 0])
            D = end
            h = RIGHT
            v = UP if C[1] > B[1] else DOWN
            l = LEFT
            r = bend_radius
            A2 = B - h * r
            B2 = B + v * r
            C1 = C - v * r
            C2 = C + l * r
            seg1 = Line(A, A2)
            curve1 = CubicBezier(A2, A2 + h * r, B2 - v * r, B2)
            seg2 = Line(B2, C1)
            curve2 = CubicBezier(C1, C1 + v * r, C2 - l * r, C2)
            seg3 = Line(C2, D)
            path = VGroup(seg1, curve1, seg2, curve2, seg3)
            path.set_stroke(res_color, width=stroke_w)
            path.set_z_index(3)
            tip = ArrowTriangleFilledTip(color=res_color)
            tip.scale(0.35)
            tip.rotate(angle_of_vector(D - C2) + PI)
            tip.shift(D - tip.get_tip_point())
            tip.set_z_index(4)
            return VGroup(path, tip)

        res1 = residual_ortho(arrows[1], blocks[3], x_offset=1.37)
        res2 = residual_ortho(arrows[3], blocks[5], x_offset=1.37)

        # Animate
        self.play(FadeIn(title), run_time=0.5)
        self.play(
            LaggedStart(*[FadeIn(b, shift=UP * 0.05) for b in blocks], lag_ratio=0.05),
            LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.05),
            run_time=1.2,
        )
        self.play(
            Create(container),
            GrowFromCenter(brace), FadeIn(n_label),
            Create(res1), Create(res2),
            run_time=0.6,
        )
        self.wait(0.8)

        # Store references
        self.pipeline_title = title
        self.blocks = blocks
        self.arrows = arrows
        self.container = container
        self.brace = brace
        self.n_label_pip = n_label
        self.res1 = res1
        self.res2 = res2

    # ------------------------------------------------------------------
    # PHASE 2 — Zoom into MHA
    # ------------------------------------------------------------------

    def phase_2_zoom_into_mha(self):
        mha = self.blocks[2]

        # Highlight MHA
        self.play(
            mha[0].animate.set_stroke(YELLOW, width=5),
            run_time=0.5,
        )

        zoom_text = Text("Let's look inside...", font_size=24, color=ACCENT_COLOR)
        zoom_text.next_to(mha, DOWN, buff=0.6)
        self.play(Write(zoom_text), run_time=0.8)
        self.wait(0.5)

        # Fade out everything except MHA
        fade_targets = [
            self.pipeline_title, zoom_text,
            self.container, self.brace, self.n_label_pip,
            self.res1, self.res2,
        ]
        for i, b in enumerate(self.blocks):
            if i != 2:
                fade_targets.append(b)
        fade_targets.extend(self.arrows)

        self.play(*[FadeOut(m) for m in fade_targets], run_time=0.8)

        # Scale MHA and move to top
        self.play(
            mha.animate.scale(1.8).move_to(UP * 3.2),
            run_time=0.8,
        )
        self.play(
            mha[0].animate.set_stroke(ATTN_COLOR, width=3),
            run_time=0.3,
        )

        inner_title = Text(
            "Inside Multi-Head Attention",
            font_size=28, color=ACCENT_COLOR,
        )
        inner_title.to_edge(UP, buff=0.15)
        self.play(FadeIn(inner_title), run_time=0.5)

        self.mha_header = mha
        self.inner_title = inner_title

    # ------------------------------------------------------------------
    # PHASE 3 — Input X
    # ------------------------------------------------------------------

    def phase_3_input_x(self):
        x_mat = TensorMatrix(T, D_MODEL, cell_size=0.2)
        x_mat.move_to(DOWN * 0.8)

        x_label = MathTex("X", font_size=28, color=ACCENT_COLOR)
        x_label.next_to(x_mat, DOWN, buff=0.15)

        brace_t = Brace(x_mat, LEFT, buff=0.1, color=GRAY_B)
        t_label = MathTex("t", font_size=18, color=GRAY_B)
        t_label.next_to(brace_t, LEFT, buff=0.05)

        brace_d = Brace(x_mat, UP, buff=0.1, color=GRAY_B)
        d_label = MathTex("d", font_size=18, color=GRAY_B)
        d_label.next_to(brace_d, UP, buff=0.05)

        eq = EquationLabel(
            r"X \in \mathbb{R}^{t \times d_{\text{model}}}",
            description="Input from previous layer",
            font_size=26,
        )
        eq.scale(0.75).to_edge(RIGHT, buff=0.6).shift(DOWN * 0.5)

        # Arrow from MHA header to X
        arr = Arrow(
            self.mha_header.get_bottom(), x_mat.get_top(),
            buff=0.15, color=GRAY_B, stroke_width=2,
        )

        self.play(GrowArrow(arr), run_time=0.4)
        self.play(
            FadeIn(x_mat, shift=UP * 0.3),
            FadeIn(x_label),
            run_time=0.6,
        )
        self.play(
            FadeIn(brace_t), FadeIn(t_label),
            FadeIn(brace_d), FadeIn(d_label),
            FadeIn(eq),
            run_time=0.6,
        )
        self.wait(0.8)

        self.x_mat = x_mat
        self.x_label = x_label
        self.x_braces = VGroup(brace_t, t_label, brace_d, d_label)
        self.x_arr = arr
        self.phase3_eq = eq

    # ------------------------------------------------------------------
    # PHASE 4 — QKV Projections
    # ------------------------------------------------------------------

    def phase_4_qkv_projections(self):
        # Clear phase 3 layout, keep X compact at top-left
        self.play(
            FadeOut(self.mha_header), FadeOut(self.inner_title),
            FadeOut(self.x_arr), FadeOut(self.x_braces),
            FadeOut(self.phase3_eq),
            run_time=0.6,
        )

        title = Text("QKV Linear Projections", font_size=26, color=ACCENT_COLOR)
        title.to_edge(UP, buff=0.25)
        self.play(
            FadeIn(title),
            self.x_mat.animate.scale(0.8).move_to(LEFT * 4.8 + DOWN * 0.3),
            self.x_label.animate.scale(0.8).move_to(LEFT * 4.8 + DOWN * 1.1),
            run_time=0.6,
        )

        proj_names = ["Q", "K", "V"]
        proj_colors = ["#42a5f5", "#66bb6a", "#ef5350"]
        results = []

        for idx, (name, color) in enumerate(zip(proj_names, proj_colors)):
            # Weight matrix
            w_mat = TensorMatrix(D_MODEL, D_MODEL, cell_size=0.12)
            w_mat.move_to(LEFT * 1.5 + DOWN * 0.3)

            w_label = MathTex(f"W_{name}", font_size=22, color=color)
            w_label.next_to(w_mat, DOWN, buff=0.1)

            times_sign = MathTex(r"\times", font_size=26, color=GRAY_B)
            times_sign.move_to(LEFT * 3.1 + DOWN * 0.3)

            eq_sign = MathTex("=", font_size=26, color=GRAY_B)
            eq_sign.move_to(RIGHT * 0.5 + DOWN * 0.3)

            # Result matrix
            r_mat = TensorMatrix(T, D_MODEL, cell_size=0.15)
            r_mat.move_to(RIGHT * 2.5 + DOWN * 0.3)

            r_label = MathTex(name, font_size=22, color=color)
            r_label.next_to(r_mat, DOWN, buff=0.1)

            # Dimension braces for result
            r_brace_t = Brace(r_mat, LEFT, buff=0.08, color=GRAY_B)
            r_t = MathTex("t", font_size=14, color=GRAY_B)
            r_t.next_to(r_brace_t, LEFT, buff=0.04)
            r_brace_d = Brace(r_mat, UP, buff=0.08, color=GRAY_B)
            r_d = MathTex("d", font_size=14, color=GRAY_B)
            r_d.next_to(r_brace_d, UP, buff=0.04)

            # Equation
            eq = EquationLabel(
                f"{name} = X W_{{{name}}}, \\quad W_{{{name}}} \\in \\mathbb{{R}}^{{d \\times d}}",
                font_size=22,
            )
            eq.scale(0.8).to_edge(RIGHT, buff=0.3).shift(DOWN * 2.3)

            # Animate
            speed = 0.5 if idx == 0 else 0.3

            self.play(
                FadeIn(w_mat, shift=UP * 0.2),
                FadeIn(w_label),
                FadeIn(times_sign),
                run_time=speed,
            )

            # Flash multiplication
            self.play(
                self.x_mat.border.animate.set_stroke(YELLOW, width=3),
                w_mat.border.animate.set_stroke(YELLOW, width=3),
                run_time=0.2,
            )
            self.play(
                self.x_mat.border.animate.set_stroke(WHITE, width=2),
                w_mat.border.animate.set_stroke(WHITE, width=2),
                run_time=0.2,
            )

            self.play(
                FadeIn(eq_sign),
                FadeIn(r_mat, shift=RIGHT * 0.2),
                FadeIn(r_label),
                FadeIn(r_brace_t), FadeIn(r_t),
                FadeIn(r_brace_d), FadeIn(r_d),
                FadeIn(eq),
                run_time=speed,
            )
            self.wait(0.3)

            # Store result for later
            results.append((r_mat.copy(), r_label.copy(), color))

            # Clean projection visuals (keep equation briefly)
            self.play(
                FadeOut(w_mat), FadeOut(w_label),
                FadeOut(times_sign), FadeOut(eq_sign),
                FadeOut(r_mat), FadeOut(r_label),
                FadeOut(r_brace_t), FadeOut(r_t),
                FadeOut(r_brace_d), FadeOut(r_d),
                FadeOut(eq),
                run_time=0.3,
            )

        # Show all three side by side
        self.play(
            FadeOut(self.x_mat), FadeOut(self.x_label),
            run_time=0.3,
        )

        combined_eq = EquationLabel(
            [
                r"Q = XW_Q",
                r"K = XW_K",
                r"V = XW_V",
            ],
            font_size=22,
        )
        combined_eq.scale(0.8).to_edge(UP, buff=0.25).shift(RIGHT * 3.5)

        q_mat, q_lbl, _ = results[0]
        k_mat, k_lbl, _ = results[1]
        v_mat, v_lbl, _ = results[2]

        positions = [LEFT * 3.5, ORIGIN, RIGHT * 3.5]
        y_pos = DOWN * 0.5

        for mat, lbl, pos in zip(
            [q_mat, k_mat, v_mat],
            [q_lbl, k_lbl, v_lbl],
            positions,
        ):
            mat.move_to(pos + y_pos)
            lbl.next_to(mat, DOWN, buff=0.12)

        self.play(Transform(title, title.copy()), run_time=0.01)  # keep title
        self.play(
            FadeIn(q_mat), FadeIn(q_lbl),
            FadeIn(k_mat), FadeIn(k_lbl),
            FadeIn(v_mat), FadeIn(v_lbl),
            FadeIn(combined_eq),
            run_time=0.8,
        )
        self.wait(0.8)

        self.q_mat = q_mat
        self.k_mat = k_mat
        self.v_mat = v_mat
        self.q_lbl = q_lbl
        self.k_lbl = k_lbl
        self.v_lbl = v_lbl
        self.phase4_title = title
        self.phase4_eq = combined_eq

    # ------------------------------------------------------------------
    # PHASE 5 — Head splitting
    # ------------------------------------------------------------------

    def phase_5_head_splitting(self):
        self.play(
            FadeOut(self.phase4_title),
            FadeOut(self.phase4_eq),
            run_time=0.4,
        )

        title = Text("Head Splitting", font_size=26, color=ACCENT_COLOR)
        title.to_edge(UP, buff=0.25)

        eq = EquationLabel(
            [
                r"d_k = d_{\text{model}} / h = 8 / 2 = 4",
                r"Q \to [Q_1, Q_2],\; Q_i \in \mathbb{R}^{t \times d_k}",
            ],
            font_size=22,
        )
        eq.scale(0.8).to_edge(RIGHT, buff=0.3).shift(UP * 2.5)

        self.play(FadeIn(title), FadeIn(eq), run_time=0.5)

        # For each of Q, K, V: split into two head sub-matrices
        all_heads = [[], []]  # all_heads[head_idx] = list of (mat, label)
        originals = [
            (self.q_mat, self.q_lbl, "Q"),
            (self.k_mat, self.k_lbl, "K"),
            (self.v_mat, self.v_lbl, "V"),
        ]

        for orig_mat, orig_lbl, name in originals:
            center = orig_mat.get_center()

            # Create two half-matrices
            h1 = TensorMatrix(T, D_K, cell_size=0.15)
            h2 = TensorMatrix(T, D_K, cell_size=0.15)
            h1.move_to(center)
            h2.move_to(center)

            h1_lbl = MathTex(f"{name}_1", font_size=18, color=HEAD_COLORS[0])
            h2_lbl = MathTex(f"{name}_2", font_size=18, color=HEAD_COLORS[1])

            # Dashed split line
            split_line = DashedLine(
                orig_mat.get_top(), orig_mat.get_bottom(),
                dash_length=0.08, color=YELLOW, stroke_width=2,
            )

            self.play(Create(split_line), run_time=0.3)
            self.play(
                FadeOut(orig_mat), FadeOut(orig_lbl),
                FadeIn(h1), FadeIn(h2),
                FadeOut(split_line),
                run_time=0.3,
            )

            # Slide apart
            offset = RIGHT * 0.6
            self.play(
                h1.animate.shift(-offset),
                h2.animate.shift(offset),
                run_time=0.4,
            )

            # Color borders
            h1.border.set_stroke(HEAD_COLORS[0], width=2)
            h2.border.set_stroke(HEAD_COLORS[1], width=2)

            h1_lbl.next_to(h1, DOWN, buff=0.08)
            h2_lbl.next_to(h2, DOWN, buff=0.08)
            self.play(FadeIn(h1_lbl), FadeIn(h2_lbl), run_time=0.2)

            all_heads[0].append((h1, h1_lbl))
            all_heads[1].append((h2, h2_lbl))

        self.wait(0.5)

        # Rearrange into head groups
        # Head 1: Q_1, K_1, V_1 on left; Head 2: Q_2, K_2, V_2 on right
        head_containers = []
        for h_idx in range(N_HEADS):
            items = all_heads[h_idx]
            group = VGroup(*[VGroup(m, l) for m, l in items])
            group.arrange(DOWN, buff=0.3)
            target_x = LEFT * 3.0 if h_idx == 0 else RIGHT * 3.0
            group.move_to(target_x + DOWN * 0.3)

            container = SurroundingRectangle(
                group, buff=0.2,
                stroke_color=HEAD_COLORS[h_idx], stroke_width=2,
                fill_opacity=0.05,
            )
            h_title = Text(
                f"Head {h_idx + 1}", font_size=18,
                color=HEAD_COLORS[h_idx],
            )
            h_title.next_to(container, UP, buff=0.1)
            head_containers.append(VGroup(group, container, h_title))

        self.play(
            *[FadeIn(hc) for hc in head_containers],
            *[FadeOut(m) for h in all_heads for m, l in h],
            *[FadeOut(l) for h in all_heads for m, l in h],
            run_time=1.0,
        )
        self.wait(0.5)

        self.head_containers = head_containers
        self.phase5_title = title
        self.phase5_eq = eq

    # ------------------------------------------------------------------
    # PHASE 6 — Scaled dot-product attention
    # ------------------------------------------------------------------

    def phase_6_scaled_dot_product(self):
        # Clear and set up for attention computation
        self.play(
            FadeOut(self.phase5_title), FadeOut(self.phase5_eq),
            *[FadeOut(hc) for hc in self.head_containers],
            run_time=0.6,
        )

        title = Text("Scaled Dot-Product Attention (Head 1)", font_size=24, color=HEAD_COLORS[0])
        title.to_edge(UP, buff=0.2)
        self.play(FadeIn(title), run_time=0.4)

        # Show Q_1, K_1, V_1 at the top
        q1 = TensorMatrix(T, D_K, cell_size=0.14)
        k1 = TensorMatrix(T, D_K, cell_size=0.14)
        v1 = TensorMatrix(T, D_K, cell_size=0.14)

        q1_lbl = MathTex("Q_1", font_size=18, color=HEAD_COLORS[0])
        k1_lbl = MathTex("K_1", font_size=18, color=HEAD_COLORS[0])
        v1_lbl = MathTex("V_1", font_size=18, color=HEAD_COLORS[0])

        input_group = VGroup(
            VGroup(q1, q1_lbl.next_to(q1, DOWN, buff=0.06)),
            VGroup(k1, k1_lbl.next_to(k1, DOWN, buff=0.06)),
            VGroup(v1, v1_lbl.next_to(v1, DOWN, buff=0.06)),
        )
        input_group.arrange(RIGHT, buff=0.6)
        input_group.next_to(title, DOWN, buff=0.35)

        for m in [q1, k1, v1]:
            m.border.set_stroke(HEAD_COLORS[0], width=1.5)

        self.play(FadeIn(input_group), run_time=0.5)
        self.wait(0.3)

        # ── 6a: QK^T ──
        step_label = Text("Step 1: Compute attention scores", font_size=18, color=GRAY_B)
        step_label.to_edge(LEFT, buff=0.3).shift(DOWN * 0.6)

        eq_6a = EquationLabel(
            r"S = Q_1 K_1^\top \in \mathbb{R}^{t \times t}",
            font_size=22,
        )
        eq_6a.scale(0.75).to_edge(RIGHT, buff=0.4).shift(DOWN * 0.6)

        scores_mat = AttentionMatrix(
            RAW_SCORES, cell_size=0.32, show_values=True,
            label="S (raw scores)", value_font_size=8,
        )
        scores_mat.move_to(DOWN * 2.2)

        self.play(FadeIn(step_label), FadeIn(eq_6a), run_time=0.4)

        # Flash Q and K
        self.play(
            q1.border.animate.set_stroke(YELLOW, width=3),
            k1.border.animate.set_stroke(YELLOW, width=3),
            run_time=0.25,
        )
        self.play(
            q1.border.animate.set_stroke(HEAD_COLORS[0], width=1.5),
            k1.border.animate.set_stroke(HEAD_COLORS[0], width=1.5),
            FadeIn(scores_mat, shift=UP * 0.3),
            run_time=0.5,
        )
        self.wait(0.6)

        # ── 6b: Scale ──
        new_step = Text("Step 2: Scale by sqrt(d_k)", font_size=18, color=GRAY_B)
        new_step.move_to(step_label)

        eq_6b = EquationLabel(
            [
                r"S = \frac{Q_1 K_1^\top}{\sqrt{d_k}}",
                r"\sqrt{d_k} = \sqrt{4} = 2",
            ],
            font_size=22,
        )
        eq_6b.scale(0.75).move_to(eq_6a)

        scaled_mat = AttentionMatrix(
            SCALED_SCORES, cell_size=0.32, show_values=True,
            label="S (scaled)", value_font_size=8,
        )
        scaled_mat.move_to(scores_mat.get_center())

        self.play(
            Transform(step_label, new_step),
            Transform(eq_6a, eq_6b),
            run_time=0.3,
        )
        self.play(
            Transform(scores_mat, scaled_mat),
            run_time=0.6,
        )
        self.wait(0.6)

        # ── 6c: Causal mask ──
        new_step2 = Text("Step 3: Apply causal mask", font_size=18, color=GRAY_B)
        new_step2.move_to(step_label)

        eq_6c = EquationLabel(
            [
                r"M_{ij}=\begin{cases}0,& j\le i\\ -\infty,& j>i\end{cases}",
                r"S_{\text{masked}} = S + M",
            ],
            font_size=20,
        )
        eq_6c.scale(0.75).move_to(eq_6a)

        # Show mask matrix
        mask_display = np.where(CAUSAL_MASK == 0, 0.0, CAUSAL_MASK)
        mask_mat = AttentionMatrix(
            mask_display, cell_size=0.32, show_values=True,
            label="M (causal mask)", value_font_size=8,
            low_color="#1b5e20", high_color="#1b5e20",
        )
        mask_mat.move_to(LEFT * 3.5 + DOWN * 2.2)

        plus_sign = MathTex("+", font_size=28, color=GRAY_B)
        plus_sign.move_to(LEFT * 1.2 + DOWN * 2.2)

        eq_sign_m = MathTex("=", font_size=28, color=GRAY_B)
        eq_sign_m.move_to(RIGHT * 1.2 + DOWN * 2.2)

        # Move scores to center-right, show mask on left
        masked_vals = MASKED_SCORES.copy()
        masked_mat = AttentionMatrix(
            masked_vals, cell_size=0.32, show_values=True,
            label="S_masked", value_font_size=8,
        )
        masked_mat.move_to(RIGHT * 3.5 + DOWN * 2.2)

        self.play(
            Transform(step_label, new_step2),
            Transform(eq_6a, eq_6c),
            scores_mat.animate.move_to(ORIGIN + DOWN * 2.2),
            run_time=0.4,
        )
        self.play(
            FadeIn(mask_mat, shift=RIGHT * 0.3),
            FadeIn(plus_sign),
            run_time=0.5,
        )
        self.play(
            FadeIn(eq_sign_m),
            FadeIn(masked_mat, shift=LEFT * 0.3),
            run_time=0.5,
        )
        self.wait(0.8)

        # Clean up mask visuals, keep masked result
        self.play(
            FadeOut(mask_mat), FadeOut(plus_sign),
            FadeOut(eq_sign_m), FadeOut(scores_mat),
            masked_mat.animate.move_to(DOWN * 2.2),
            run_time=0.5,
        )

        # ── 6d: Softmax ──
        new_step3 = Text("Step 4: Softmax (row-wise)", font_size=18, color=GRAY_B)
        new_step3.move_to(step_label)

        eq_6d = EquationLabel(
            [
                r"A = \text{softmax}(S_{\text{masked}})",
                r"A_{ij} = \frac{e^{S_{ij}}}{\sum_k e^{S_{ik}}}",
            ],
            font_size=22,
        )
        eq_6d.scale(0.75).move_to(eq_6a)

        attn_mat = AttentionMatrix(
            ATTN_WEIGHTS, cell_size=0.32, show_values=True,
            label="A (attention weights)",
            value_font_size=8,
            low_color="#1a237e", high_color=YELLOW,
        )
        attn_mat.move_to(masked_mat.get_center())

        self.play(
            Transform(step_label, new_step3),
            Transform(eq_6a, eq_6d),
            run_time=0.3,
        )

        # Animate row-by-row softmax transformation
        self.play(Transform(masked_mat, attn_mat), run_time=1.0)

        # Highlight rows sum to 1
        row_note = Text("Each row sums to 1", font_size=14, color=SOFTMAX_COLOR)
        row_note.next_to(masked_mat, RIGHT, buff=0.3)
        self.play(FadeIn(row_note), run_time=0.3)
        self.wait(0.8)
        self.play(FadeOut(row_note), run_time=0.2)

        # ── 6e: Weighted sum A * V ──
        new_step4 = Text("Step 5: Weighted sum with V", font_size=18, color=GRAY_B)
        new_step4.move_to(step_label)

        eq_6e = EquationLabel(
            r"\text{head}_1 = A \cdot V_1 \in \mathbb{R}^{t \times d_k}",
            font_size=22,
        )
        eq_6e.scale(0.75).move_to(eq_6a)

        # Show multiplication: A (t×t) × V_1 (t×d_k) = head_1 (t×d_k)
        times_av = MathTex(r"\times", font_size=24, color=GRAY_B)
        v1_copy = TensorMatrix(T, D_K, cell_size=0.14)
        v1_copy.border.set_stroke(HEAD_COLORS[0], width=1.5)
        v1_copy_lbl = MathTex("V_1", font_size=16, color=HEAD_COLORS[0])

        eq_av = MathTex("=", font_size=24, color=GRAY_B)
        head1_result = TensorMatrix(T, D_K, cell_size=0.15)
        head1_result.border.set_stroke(HEAD_COLORS[0], width=2)
        head1_lbl = MathTex(r"\text{head}_1", font_size=16, color=HEAD_COLORS[0])

        # Position multiplication chain
        self.play(
            Transform(step_label, new_step4),
            Transform(eq_6a, eq_6e),
            masked_mat.animate.scale(0.7).move_to(LEFT * 3 + DOWN * 2.2),
            run_time=0.4,
        )

        times_av.next_to(masked_mat, RIGHT, buff=0.2)
        v1_copy.next_to(times_av, RIGHT, buff=0.2)
        v1_copy_lbl.next_to(v1_copy, DOWN, buff=0.06)
        eq_av.next_to(v1_copy, RIGHT, buff=0.2)
        head1_result.move_to(RIGHT * 3.5 + DOWN * 2.2)
        head1_lbl.next_to(head1_result, DOWN, buff=0.08)

        self.play(
            FadeIn(times_av), FadeIn(v1_copy), FadeIn(v1_copy_lbl),
            run_time=0.3,
        )
        self.play(
            FadeIn(eq_av),
            FadeIn(head1_result, shift=LEFT * 0.2),
            FadeIn(head1_lbl),
            run_time=0.5,
        )
        self.wait(0.5)

        # Full attention equation
        full_eq = EquationLabel(
            r"\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V",
            font_size=20,
        )
        full_eq.scale(0.8).to_edge(DOWN, buff=0.15)
        self.play(FadeIn(full_eq), run_time=0.5)
        self.wait(0.8)

        # ── 6f: Head 2 (abbreviated) ──
        self.play(
            FadeOut(masked_mat), FadeOut(times_av),
            FadeOut(v1_copy), FadeOut(v1_copy_lbl),
            FadeOut(eq_av), FadeOut(step_label), FadeOut(eq_6a),
            head1_result.animate.move_to(LEFT * 2.5 + DOWN * 2.0),
            head1_lbl.animate.move_to(LEFT * 2.5 + DOWN * 2.7),
            run_time=0.5,
        )

        head2_note = Text(
            "Same process for Head 2...",
            font_size=18, color=HEAD_COLORS[1],
        )
        head2_note.move_to(RIGHT * 1.5 + DOWN * 1.2)

        head2_result = TensorMatrix(T, D_K, cell_size=0.15)
        head2_result.border.set_stroke(HEAD_COLORS[1], width=2)
        head2_result.move_to(RIGHT * 2.5 + DOWN * 2.0)
        head2_lbl = MathTex(r"\text{head}_2", font_size=16, color=HEAD_COLORS[1])
        head2_lbl.next_to(head2_result, DOWN, buff=0.08)

        self.play(Write(head2_note), run_time=0.5)
        self.play(
            FadeIn(head2_result, scale=0.8),
            FadeIn(head2_lbl),
            run_time=0.5,
        )
        self.wait(0.5)
        self.play(FadeOut(head2_note), run_time=0.3)

        # Store
        self.head1_result = head1_result
        self.head1_lbl = head1_lbl
        self.head2_result = head2_result
        self.head2_lbl = head2_lbl
        self.input_group = input_group
        self.phase6_title = title
        self.phase6_full_eq = full_eq

    # ------------------------------------------------------------------
    # PHASE 7 — Concatenate heads
    # ------------------------------------------------------------------

    def phase_7_concatenate_heads(self):
        self.play(
            FadeOut(self.input_group),
            FadeOut(self.phase6_title),
            FadeOut(self.phase6_full_eq),
            run_time=0.5,
        )

        title = Text("Concatenate Heads", font_size=26, color=ACCENT_COLOR)
        title.to_edge(UP, buff=0.25)

        eq = EquationLabel(
            [
                r"\text{Concat}(\text{head}_1, \text{head}_2) \in \mathbb{R}^{t \times d}",
                r"d = h \cdot d_k = 2 \times 4 = 8",
            ],
            font_size=22,
        )
        eq.scale(0.8).to_edge(RIGHT, buff=0.5).shift(UP * 1.5)

        self.play(FadeIn(title), FadeIn(eq), run_time=0.5)

        # Animate heads sliding together
        center_x = 0
        gap = 0.05  # small gap initially, then close

        self.play(
            self.head1_result.animate.move_to(
                LEFT * (self.head1_result.width / 2 + gap) + DOWN * 0.5
            ),
            self.head1_lbl.animate.move_to(LEFT * 2 + DOWN * 1.5),
            self.head2_result.animate.move_to(
                RIGHT * (self.head2_result.width / 2 + gap) + DOWN * 0.5
            ),
            self.head2_lbl.animate.move_to(RIGHT * 2 + DOWN * 1.5),
            run_time=0.6,
        )

        # Slide together
        self.play(
            self.head1_result.animate.shift(RIGHT * gap),
            self.head2_result.animate.shift(LEFT * gap),
            run_time=0.4,
        )

        # Create combined matrix to replace them
        concat_mat = TensorMatrix(T, D_MODEL, cell_size=0.18)
        concat_center = (
            self.head1_result.get_center() + self.head2_result.get_center()
        ) / 2
        concat_mat.move_to(concat_center)

        concat_lbl = MathTex(r"\text{Concat}", font_size=20, color=ACCENT_COLOR)
        concat_lbl.next_to(concat_mat, DOWN, buff=0.12)

        brace_t = Brace(concat_mat, LEFT, buff=0.1, color=GRAY_B)
        t_label = MathTex("t", font_size=16, color=GRAY_B)
        t_label.next_to(brace_t, LEFT, buff=0.04)

        brace_d = Brace(concat_mat, UP, buff=0.1, color=GRAY_B)
        d_label = MathTex("d", font_size=16, color=GRAY_B)
        d_label.next_to(brace_d, UP, buff=0.04)

        self.play(
            FadeOut(self.head1_result), FadeOut(self.head2_result),
            FadeOut(self.head1_lbl), FadeOut(self.head2_lbl),
            FadeIn(concat_mat),
            FadeIn(concat_lbl),
            FadeIn(brace_t), FadeIn(t_label),
            FadeIn(brace_d), FadeIn(d_label),
            run_time=0.6,
        )
        self.wait(0.5)

        self.concat_mat = concat_mat
        self.concat_lbl = concat_lbl
        self.concat_braces = VGroup(brace_t, t_label, brace_d, d_label)
        self.phase7_title = title
        self.phase7_eq = eq

    # ------------------------------------------------------------------
    # PHASE 8 — Output projection
    # ------------------------------------------------------------------

    def phase_8_output_projection(self):
        self.play(
            FadeOut(self.phase7_title),
            FadeOut(self.phase7_eq),
            run_time=0.4,
        )

        title = Text("Output Projection", font_size=26, color=ACCENT_COLOR)
        title.to_edge(UP, buff=0.25)
        self.play(FadeIn(title), run_time=0.3)

        # Move concat to left
        self.play(
            self.concat_mat.animate.move_to(LEFT * 4 + DOWN * 0.3),
            self.concat_lbl.animate.move_to(LEFT * 4 + DOWN * 1.2),
            self.concat_braces.animate.shift(LEFT * 4 + UP * 0.2),
            run_time=0.5,
        )
        # Fade braces to simplify
        self.play(FadeOut(self.concat_braces), run_time=0.2)

        # W_O
        w_o = TensorMatrix(D_MODEL, D_MODEL, cell_size=0.12)
        w_o.move_to(LEFT * 1 + DOWN * 0.3)
        w_o_lbl = MathTex("W_O", font_size=20, color=ACCENT_COLOR)
        w_o_lbl.next_to(w_o, DOWN, buff=0.1)

        times_sign = MathTex(r"\times", font_size=26, color=GRAY_B)
        times_sign.move_to(LEFT * 2.4 + DOWN * 0.3)

        eq_sign = MathTex("=", font_size=26, color=GRAY_B)
        eq_sign.move_to(RIGHT * 1 + DOWN * 0.3)

        # Result
        output_mat = TensorMatrix(T, D_MODEL, cell_size=0.18)
        output_mat.move_to(RIGHT * 3.5 + DOWN * 0.3)
        output_lbl = MathTex(r"\text{MHA}(X)", font_size=20, color=ACCENT_COLOR)
        output_lbl.next_to(output_mat, DOWN, buff=0.12)

        out_brace_t = Brace(output_mat, LEFT, buff=0.1, color=GRAY_B)
        out_t = MathTex("t", font_size=16, color=GRAY_B)
        out_t.next_to(out_brace_t, LEFT, buff=0.04)
        out_brace_d = Brace(output_mat, UP, buff=0.1, color=GRAY_B)
        out_d = MathTex("d", font_size=16, color=GRAY_B)
        out_d.next_to(out_brace_d, UP, buff=0.04)

        self.play(
            FadeIn(times_sign),
            FadeIn(w_o, shift=UP * 0.2), FadeIn(w_o_lbl),
            run_time=0.5,
        )

        # Flash multiplication
        self.play(
            self.concat_mat.border.animate.set_stroke(YELLOW, width=3),
            w_o.border.animate.set_stroke(YELLOW, width=3),
            run_time=0.25,
        )
        self.play(
            self.concat_mat.border.animate.set_stroke(WHITE, width=2),
            w_o.border.animate.set_stroke(WHITE, width=2),
            run_time=0.25,
        )

        self.play(
            FadeIn(eq_sign),
            FadeIn(output_mat, shift=LEFT * 0.3),
            FadeIn(output_lbl),
            FadeIn(out_brace_t), FadeIn(out_t),
            FadeIn(out_brace_d), FadeIn(out_d),
            run_time=0.6,
        )
        self.wait(0.5)

        # Equation for W_O
        eq_wo = EquationLabel(
            r"\text{MHA}(X) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)\,W_O",
            font_size=20,
        )
        eq_wo.scale(0.8).to_edge(DOWN, buff=1.2)
        self.play(FadeIn(eq_wo), run_time=0.5)
        self.wait(0.5)

        # Final summary equation
        final_eq = EquationLabel(
            [
                r"\text{MHA}(X) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)\,W_O",
                r"\text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^\top}{\sqrt{d_k}} + M\right) V_i",
                r"Q=XW_Q,\; K=XW_K,\; V=XW_V",
            ],
            font_size=20,
        )
        final_eq.scale(0.85).move_to(DOWN * 2.5)

        self.play(Transform(eq_wo, final_eq), run_time=0.8)
        self.wait(1.5)

        # Final fade
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.5)
