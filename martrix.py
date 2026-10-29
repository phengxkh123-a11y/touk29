#python -m streamlit run martrix.py
from fractions import Fraction

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ມາຕຣິກ - ຄະນິດສາດ", page_icon="🔢", layout="wide")


# ============================================================
# ຟັງຊັນຊ່ວຍ: ການສະແດງຜົນ
# ============================================================
def _fmt(x):
    """ຈັດຮູບແບບຕົວເລກ (ຮອງຮັບຈຳນວນຊັບຊ້ອນ)"""
    if isinstance(x, Fraction):
        return fl(x)
    if isinstance(x, complex) or np.iscomplexobj(x):
        x = complex(x)
        if abs(x.imag) < 1e-10:
            x = x.real
        else:
            return f"{x.real:.3f}{x.imag:+.3f}i"
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.4f}"


def F(x):
    """ແປງຕົວເລກເປັນເສດສ່ວນ (Fraction) ເພື່ອໃຫ້ການຄິດໄລ່ແນ່ນອນ ບໍ່ມີຄ່າຄາດເຄື່ອນ"""
    return Fraction(float(x)).limit_denominator(10000)


def fl(x):
    """ຕົວເລກ / ເສດສ່ວນ → LaTeX"""
    if isinstance(x, Fraction):
        if x.denominator == 1:
            return str(x.numerator)
        sign = "-" if x < 0 else ""
        return rf"{sign}\frac{{{abs(x.numerator)}}}{{{x.denominator}}}"
    return _fmt(x)


def to_latex(M):
    """numpy array → LaTeX bmatrix"""
    M = np.atleast_2d(M)
    rows = [" & ".join(_fmt(x) for x in row) for row in M]
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"


def str_latex(rows):
    """ມາຕຣິກຂອງ string (ສູດ/ການຄິດໄລ່ໃນແຕ່ລະຊ່ອງ) → LaTeX"""
    body = r" \\ ".join(" & ".join(r) for r in rows)
    return r"\begin{bmatrix}" + body + r"\end{bmatrix}"


def aug_latex(M, split):
    """ມາຕຣິກຂະຫຍາຍ [ A | b ] ພ້ອມເສັ້ນຂັ້ນ"""
    cols = len(M[0])
    spec = "c" * split + ("|" + "c" * (cols - split) if split < cols else "")
    body = r" \\ ".join(" & ".join(fl(x) for x in row) for row in M)
    return rf"\left[\begin{{array}}{{{spec}}}{body}\end{{array}}\right]"


def Rn(i):
    """ຊື່ແຖວ R_i (i ເລີ່ມຈາກ 0)"""
    return f"R_{{{i + 1}}}"


def matrix_input(label, rows, cols, key, default="identity"):
    """ຕາຕະລາງໃຫ້ຜູ້ໃຊ້ແກ້ໄຂຄ່າຂອງມາຕຣິກ"""
    if default == "identity":
        init = np.eye(rows, cols)
    else:
        rng = np.random.default_rng(42)
        init = rng.integers(-5, 10, size=(rows, cols)).astype(float)
    df = pd.DataFrame(
        init,
        columns=[f"c{j + 1}" for j in range(cols)],
        index=[f"r{i + 1}" for i in range(rows)],
    )
    # ໃສ່ຂະໜາດໃນ key ເພື່ອໃຫ້ຕາຕະລາງຣີເຊັດເມື່ອປ່ຽນຂະໜາດ
    edited = st.data_editor(df, key=f"{key}_{rows}x{cols}", use_container_width=True)
    return edited.to_numpy(dtype=float)


def show(title, M):
    st.markdown(f"**{title}**")
    st.latex(to_latex(M))


def show_steps(steps, split):
    """ສະແດງລຳດັບຂັ້ນຕອນ: ຄຳອະທິບາຍ + ມາຕຣິກຫຼັງແຕ່ລະຂັ້ນ"""
    for k, (desc, M) in enumerate(steps):
        st.markdown(f"**ຂັ້ນທີ {k}.** {desc}" if k else f"**ເລີ່ມຕົ້ນ:** {desc}")
        st.latex(aug_latex(M, split))


# ============================================================
# ຟັງຊັນຊ່ວຍ: ສູດຄິດໄລ່ແບບມີຂັ້ນຕອນ
# ============================================================
def gauss_jordan(A, split):
    """
    Gauss-Jordan elimination ໃສ່ມາຕຣິກ A (ຈະເປັນມາຕຣິກຂະຫຍາຍກໍໄດ້)
    split = ຈຳນວນຖັນທີ່ເປັນສ່ວນສຳປະສິດ (ໃຊ້ຫາ pivot)
    ສົ່ງຄືນ (ມາຕຣິກຜົນ, ລາຍການຂັ້ນຕອນ, ລາຍການຖັນ pivot)
    """
    M = [[F(x) for x in row] for row in np.atleast_2d(A)]
    rows = len(M)
    steps = [("ມາຕຣິກຕັ້ງຕົ້ນ", [r[:] for r in M])]
    r = 0
    pivots = []
    for c in range(split):
        if r >= rows:
            break
        p = next((i for i in range(r, rows) if M[i][c] != 0), None)
        if p is None:
            continue  # ຖັນນີ້ບໍ່ມີ pivot
        if p != r:
            M[r], M[p] = M[p], M[r]
            steps.append(
                (f"ສະຫຼັບແຖວ ${Rn(r)} \\leftrightarrow {Rn(p)}$ (ເພາະ pivot ຖັນ {c + 1} ເປັນ 0)",
                 [x[:] for x in M])
            )
        if M[r][c] != 1:
            f = M[r][c]
            M[r] = [x / f for x in M[r]]
            steps.append(
                (f"${Rn(r)} \\to {Rn(r)} \\div ({fl(f)})$ ເພື່ອໃຫ້ pivot ຖັນ {c + 1} ເທົ່າ 1",
                 [x[:] for x in M])
            )
        for i in range(rows):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
                steps.append(
                    (f"${Rn(i)} \\to {Rn(i)} - ({fl(f)})\\cdot {Rn(r)}$ (ກຳຈັດຄ່າໃນຖັນ {c + 1} ໃຫ້ເປັນ 0)",
                     [x[:] for x in M])
                )
        pivots.append(c)
        r += 1
    return M, steps, pivots


def det_by_elimination(A):
    """
    ຫາ det ໂດຍການເຮັດໃຫ້ເປັນມາຕຣິກສາມແຈເທິງ (ບໍ່ຕ້ອງປັບ pivot ເປັນ 1)
    ສົ່ງຄືນ (det, ຂັ້ນຕອນ, ມາຕຣິກສາມແຈ, sign)
    """
    M = [[F(x) for x in row] for row in A]
    n = len(M)
    sign = 1
    steps = [("ມາຕຣິກຕັ້ງຕົ້ນ", [r[:] for r in M])]
    for c in range(n):
        p = next((i for i in range(c, n) if M[i][c] != 0), None)
        if p is None:
            return Fraction(0), steps, M, sign
        if p != c:
            M[c], M[p] = M[p], M[c]
            sign = -sign
            steps.append(
                (f"ສະຫຼັບແຖວ ${Rn(c)} \\leftrightarrow {Rn(p)}$ (det ປ່ຽນເຄື່ອງໝາຍ)", [x[:] for x in M])
            )
        for i in range(c + 1, n):
            if M[i][c] != 0:
                f = M[i][c] / M[c][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[c])]
                steps.append(
                    (f"${Rn(i)} \\to {Rn(i)} - ({fl(f)})\\cdot {Rn(c)}$ (det ບໍ່ປ່ຽນ)", [x[:] for x in M])
                )
    prod = Fraction(1)
    for i in range(n):
        prod *= M[i][i]
    return sign * prod, steps, M, sign


def poly_latex(coefs):
    """ສຳປະສິດ (ກຳລັງສູງ → ຕ່ຳ) → ພຫຸນາມ λ ໃນ LaTeX"""
    n = len(coefs) - 1
    terms = []
    for i, c in enumerate(coefs):
        p = n - i
        c = float(np.real(c))
        if abs(c) < 1e-10:
            continue
        mag = abs(c)
        cs = "" if (abs(mag - 1) < 1e-9 and p > 0) else _fmt(mag)
        var = "" if p == 0 else (r"\lambda" if p == 1 else rf"\lambda^{{{p}}}")
        terms.append(("-" if c < 0 else "+", cs + var))
    out = ""
    for k, (sg, t) in enumerate(terms):
        out += (("-" if sg == "-" else "") + t) if k == 0 else f" {sg} {t}"
    return out or "0"


# ============================================================
# ສ່ວນຫົວ
# ============================================================
st.title("🔢 ເຄື່ອງຄິດເລກມາຕຣິກ")
st.caption("ວິຊາຄະນິດສາດ: ພີຊະຄະນິດເສັ້ນຊື່ (Linear Algebra) ດ້ວຍ Python + Streamlit — ພ້ອມວິທີຄິດໄລ່ທີລະຂັ້ນຕອນ")

mode = st.sidebar.radio(
    "ເລືອກໝວດການຄິດໄລ່",
    [
        "ບວກ / ລົບ / ຄູນ ມາຕຣິກ",
        "ຄຸນສົມບັດຂອງມາຕຣິກດຽວ",
        "ແກ້ລະບົບສົມຜົນເສັ້ນຊື່ (Ax = b)",
        "ຄ່າເຈາະຈົງ (Eigenvalues)",
    ],
)
open_all = st.sidebar.checkbox("ເປີດວິທີຄິດໄລ່ທັງໝົດໄວ້ເລີຍ", value=True)

# ------------------------------------------------------------
# 1) ບວກ ລົບ ຄູນ
# ------------------------------------------------------------
if mode == "ບວກ / ລົບ / ຄູນ ມາຕຣິກ":
    st.header("ບວກ / ລົບ / ຄູນ ມາຕຣິກ")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ມາຕຣິກ A")
        ra = st.number_input("ຈຳນວນແຖວ A (m)", 1, 6, 2, key="ra")
        ca = st.number_input("ຈຳນວນຖັນ A (n)", 1, 6, 2, key="ca")
        A = matrix_input("A", int(ra), int(ca), "A", default="random")
    with c2:
        st.subheader("ມາຕຣິກ B")
        rb = st.number_input("ຈຳນວນແຖວ B", 1, 6, 2, key="rb")
        cb = st.number_input("ຈຳນວນຖັນ B (p)", 1, 6, 2, key="cb")
        B = matrix_input("B", int(rb), int(cb), "B", default="random")

    st.divider()
    op = st.selectbox("ເລືອກການຄຳນວນ", ["A + B", "A - B", "A × B", "k × A"])

    if op in ("A + B", "A - B"):
        if A.shape != B.shape:
            st.error("ບວກ/ລົບ ໄດ້ສະເພາະມາຕຣິກທີ່ມີຂະໜາດເທົ່າກັນ")
        else:
            add = op == "A + B"
            R = A + B if add else A - B
            show(f"ຜົນລັບ {op}", R)
            with st.expander("📖 ວິທີຄິດໄລ່", expanded=open_all):
                word = "ບວກ" if add else "ລົບ"
                sym = "+" if add else "-"
                st.markdown(
                    f"**ຫຼັກການ:** {word}ມາຕຣິກ = {word}ຄ່າໃນຕຳແໜ່ງດຽວກັນ ທີລະຄູ່ "
                    f"(ຂະໜາດຕ້ອງເທົ່າກັນ)"
                )
                st.latex(rf"(A {sym} B)_{{ij}} = a_{{ij}} {sym} b_{{ij}}")
                st.markdown("**ຂັ້ນທີ 1:** ແທນຄ່າຂອງແຕ່ລະຊ່ອງ")
                expr = [
                    [f"({_fmt(A[i, j])}) {sym} ({_fmt(B[i, j])})" for j in range(A.shape[1])]
                    for i in range(A.shape[0])
                ]
                st.latex(str_latex(expr))
                st.markdown("**ຂັ້ນທີ 2:** ຄິດໄລ່ໃນແຕ່ລະຊ່ອງ")
                st.latex(to_latex(R))

    elif op == "A × B":
        if A.shape[1] != B.shape[0]:
            st.error(
                f"ຄູນບໍ່ໄດ້: ຈຳນວນຖັນຂອງ A ({A.shape[1]}) "
                f"ຕ້ອງເທົ່າກັບຈຳນວນແຖວຂອງ B ({B.shape[0]})"
            )
        else:
            R = A @ B
            show(f"ຜົນຄູນ A × B (ຂະໜາດ {R.shape[0]}×{R.shape[1]})", R)
            with st.expander("📖 ວິທີຄິດໄລ່ (ທຸກຊ່ອງ)", expanded=open_all):
                st.markdown(
                    f"**ຂັ້ນທີ 1: ກວດຂະໜາດ** — A ເປັນ {A.shape[0]}×{A.shape[1]}, "
                    f"B ເປັນ {B.shape[0]}×{B.shape[1]}. ຖັນຂອງ A = ແຖວຂອງ B = {A.shape[1]} ✔ "
                    f"ຜົນລັບຈຶ່ງມີຂະໜາດ {R.shape[0]}×{R.shape[1]}"
                )
                st.markdown("**ຂັ້ນທີ 2: ສູດ** — ຊ່ອງ (i, j) = ແຖວທີ i ຂອງ A ຄູນແບບຈຸດ (dot) ກັບຖັນທີ j ຂອງ B")
                st.latex(r"c_{ij} = \sum_{k} a_{ik}\, b_{kj} = a_{i1}b_{1j} + a_{i2}b_{2j} + \cdots")
                st.markdown("**ຂັ້ນທີ 3: ຄິດໄລ່ທຸກຊ່ອງ**")
                for i in range(R.shape[0]):
                    for j in range(R.shape[1]):
                        terms = " + ".join(
                            f"({_fmt(A[i, k])})({_fmt(B[k, j])})" for k in range(A.shape[1])
                        )
                        st.latex(rf"c_{{{i + 1}{j + 1}}} = {terms} = {_fmt(R[i, j])}")
                st.markdown("**ຂັ້ນທີ 4: ລວມເປັນມາຕຣິກຜົນລັບ**")
                st.latex(to_latex(R))

    else:
        k = st.number_input("ຄ່າຄົງທີ່ k", value=2.0)
        show(f"ຜົນລັບ {k} × A", k * A)
        with st.expander("📖 ວິທີຄິດໄລ່", expanded=open_all):
            st.markdown("**ຫຼັກການ:** ຄູນຄ່າຄົງທີ່ k ເຂົ້າກັບທຸກຊ່ອງຂອງມາຕຣິກ")
            st.latex(r"(kA)_{ij} = k \cdot a_{ij}")
            expr = [
                [f"({_fmt(k)})({_fmt(A[i, j])})" for j in range(A.shape[1])]
                for i in range(A.shape[0])
            ]
            st.latex(str_latex(expr))
            st.latex(to_latex(k * A))

# ------------------------------------------------------------
# 2) ຄຸນສົມບັດຂອງມາຕຣິກດຽວ
# ------------------------------------------------------------
elif mode == "ຄຸນສົມບັດຂອງມາຕຣິກດຽວ":
    st.header("ຄຸນສົມບັດຂອງມາຕຣິກ")
    n = st.slider("ຂະໜາດມາຕຣິກຈະຕຸລັດ n × n", 2, 5, 3)
    A = matrix_input("A", n, n, "single", default="random")
    show("ມາຕຣິກ A", A)

    det_f, det_steps, tri, sign = det_by_elimination(A)
    _, rref_steps_list, pivots = gauss_jordan(A, n)
    rank = len(pivots)

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("ດີເຕີມີແນນ det(A)", _fmt(float(det_f)))
    col2.metric("ຣັງ (Rank)", rank)
    col3.metric("ເທຣສ (Trace)", _fmt(np.trace(A)))

    # ---------- Determinant ----------
    st.subheader("1) ດີເຕີມີແນນ (Determinant)")
    with st.expander("📖 ວິທີຄິດໄລ່ det(A)", expanded=open_all):
        if n == 2:
            a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
            st.markdown("**ສູດສຳລັບມາຕຣິກ 2×2:** ຜົນຄູນເສັ້ນຂວາງຫຼັກ ລົບ ຜົນຄູນເສັ້ນຂວາງຮອງ")
            st.latex(r"\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc")
            st.latex(
                rf"\det(A) = ({_fmt(a)})({_fmt(d)}) - ({_fmt(b)})({_fmt(c)}) "
                rf"= {_fmt(a * d)} - ({_fmt(b * c)}) = {_fmt(a * d - b * c)}"
            )
        elif n == 3:
            m = A
            st.markdown("**ວິທີ Sarrus ສຳລັບມາຕຣິກ 3×3:** ບວກ 3 ເສັ້ນຂວາງຫຼັກ ແລ້ວລົບ 3 ເສັ້ນຂວາງຮອງ")
            st.latex(
                r"\det(A) = a_{11}a_{22}a_{33} + a_{12}a_{23}a_{31} + a_{13}a_{21}a_{32}"
                r" - a_{13}a_{22}a_{31} - a_{11}a_{23}a_{32} - a_{12}a_{21}a_{33}"
            )
            p = [
                m[0, 0] * m[1, 1] * m[2, 2], m[0, 1] * m[1, 2] * m[2, 0], m[0, 2] * m[1, 0] * m[2, 1],
                m[0, 2] * m[1, 1] * m[2, 0], m[0, 0] * m[1, 2] * m[2, 1], m[0, 1] * m[1, 0] * m[2, 2],
            ]
            g = _fmt
            st.latex(
                rf"= ({g(m[0,0])})({g(m[1,1])})({g(m[2,2])}) + ({g(m[0,1])})({g(m[1,2])})({g(m[2,0])}) "
                rf"+ ({g(m[0,2])})({g(m[1,0])})({g(m[2,1])}) - ({g(m[0,2])})({g(m[1,1])})({g(m[2,0])}) "
                rf"- ({g(m[0,0])})({g(m[1,2])})({g(m[2,1])}) - ({g(m[0,1])})({g(m[1,0])})({g(m[2,2])})"
            )
            st.latex(
                rf"= {g(p[0])} + {g(p[1])} + {g(p[2])} - ({g(p[3])}) - ({g(p[4])}) - ({g(p[5])}) "
                rf"= {g(p[0] + p[1] + p[2] - p[3] - p[4] - p[5])}"
            )
        st.markdown(
            "**ວິທີ Row Reduction (ໃຊ້ໄດ້ທຸກຂະໜາດ):** ເຮັດໃຫ້ເປັນມາຕຣິກສາມແຈເທິງ "
            "— ຖ້າສະຫຼັບແຖວ det ປ່ຽນເຄື່ອງໝາຍ, ຖ້າເອົາແຖວໜຶ່ງລົບດ້ວຍຜົນຄູນຂອງອີກແຖວ det ບໍ່ປ່ຽນ"
        )
        show_steps(det_steps, n)
        diag_txt = " \\times ".join(f"({fl(tri[i][i])})" for i in range(n))
        st.markdown("**ສະຫຼຸບ:** det = (−1)^{ຈຳນວນຄັ້ງສະຫຼັບແຖວ} × ຜົນຄູນເສັ້ນຂວາງຫຼັກ")
        if det_f == 0:
            st.latex(r"\det(A) = 0 \quad(\text{ມີ pivot ເປັນ } 0)")
        else:
            st.latex(rf"\det(A) = ({sign:+d}) \times {diag_txt} = {fl(det_f)}")

    # ---------- Rank ----------
    st.subheader("2) ຣັງ (Rank)")
    with st.expander("📖 ວິທີຫາ Rank", expanded=open_all):
        st.markdown("ເຮັດ Row Reduction ຈົນໄດ້ຮູບແບບຂັ້ນໄດ (RREF) ແລ້ວ **ນັບຈຳນວນ pivot (ຕົວນຳ 1)**")
        show_steps(rref_steps_list, n)
        st.latex(rf"\text{{rank}}(A) = {rank} \quad(\text{{pivot ຢູ່ຖັນ }} {', '.join(str(p + 1) for p in pivots) or '-'})")

    # ---------- Trace ----------
    st.subheader("3) ເທຣສ (Trace)")
    with st.expander("📖 ວິທີຄິດໄລ່ Trace", expanded=open_all):
        st.markdown("Trace = ຜົນບວກຂອງຄ່າເທິງເສັ້ນຂວາງຫຼັກ")
        diag = " + ".join(f"({_fmt(A[i, i])})" for i in range(n))
        st.latex(rf"\operatorname{{tr}}(A) = {diag} = {_fmt(np.trace(A))}")

    # ---------- Transpose ----------
    st.subheader("4) ທຣານສະໂພສ (Transpose)")
    show("Aᵀ", A.T)
    with st.expander("📖 ວິທີຄິດໄລ່", expanded=open_all):
        st.markdown("ສະຫຼັບແຖວເປັນຖັນ: ແຖວທີ i ຂອງ A ກາຍເປັນຖັນທີ i ຂອງ Aᵀ")
        st.latex(r"(A^T)_{ij} = a_{ji}")
        for i in range(n):
            row_txt = ", ".join(_fmt(x) for x in A[i, :])
            st.markdown(f"- ແຖວທີ {i + 1} ຂອງ A = ({row_txt}) → ຖັນທີ {i + 1} ຂອງ Aᵀ")

    # ---------- Inverse ----------
    st.subheader("5) ມາຕຣິກຜັນກັບ (Inverse)")
    if det_f == 0:
        st.warning("det(A) = 0 ຈຶ່ງບໍ່ມີມາຕຣິກຜັນກັບ (singular matrix)")
    else:
        inv = np.linalg.inv(A)
        show("ມາຕຣິກຜັນກັບ A⁻¹", inv)
        with st.expander("📖 ວິທີຄິດໄລ່ A⁻¹", expanded=open_all):
            st.markdown("**ເງື່ອນໄຂ:** ມີ A⁻¹ ໄດ້ກໍຕໍ່ເມື່ອ det(A) ≠ 0 ")
            st.latex(rf"\det(A) = {fl(det_f)} \neq 0 \;\Rightarrow\; \text{{ມີ }} A^{{-1}}")
            if n == 2:
                a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
                st.markdown("**ສູດລັດສຳລັບ 2×2:**")
                st.latex(
                    r"A^{-1} = \frac{1}{ad-bc}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}"
                )
                st.latex(
                    rf"A^{{-1}} = \frac{{1}}{{{fl(det_f)}}}"
                    + to_latex(np.array([[d, -b], [-c, a]]))
                )
            st.markdown(
                "**ວິທີ Gauss-Jordan:** ຂຽນ [ A | I ] ແລ້ວເຮັດ Row Reduction "
                "ຈົນດ້ານຊ້າຍເປັນ I → ດ້ານຂວາຈະກາຍເປັນ A⁻¹"
            )
            aug0 = np.hstack([A, np.eye(n)])
            _, inv_steps, _ = gauss_jordan(aug0, n)
            show_steps(inv_steps, n)
        show("ກວດສອບ: A × A⁻¹ (ຄວນເປັນມາຕຣິກຫົວໜ່ວຍ)", np.round(A @ inv, 10))

    st.info(
        f"ປະເພດ: "
        f"{'ສົມມາຕ (symmetric)' if np.allclose(A, A.T) else 'ບໍ່ສົມມາຕ'} | "
        f"{'ຫົວໜ່ວຍ (identity)' if np.allclose(A, np.eye(n)) else 'ບໍ່ແມ່ນມາຕຣິກຫົວໜ່ວຍ'}"
    )

# ------------------------------------------------------------
# 3) ແກ້ລະບົບສົມຜົນ
# ------------------------------------------------------------
elif mode == "ແກ້ລະບົບສົມຜົນເສັ້ນຊື່ (Ax = b)":
    st.header("ແກ້ລະບົບສົມຜົນເສັ້ນຊື່  Ax = b")
    n = st.slider("ຈຳນວນຕົວແປ n", 2, 5, 2)

    c1, c2 = st.columns([n + 1, 1])
    with c1:
        st.subheader("ມາຕຣິກສຳປະສິດ A")
        A = matrix_input("A", n, n, "sysA", default="random")
    with c2:
        st.subheader("ເວັກເຕີ b")
        b = matrix_input("b", n, 1, "sysb", default="random").flatten()

    names = [f"x{i + 1}" for i in range(n)]
    eqs = []
    for i in range(n):
        left = " + ".join(f"{_fmt(A[i, j])}{names[j]}" for j in range(n))
        eqs.append(f"{left} = {_fmt(b[i])}")
    st.markdown("**ລະບົບສົມຜົນ:**")
    st.code("\n".join(eqs))

    # Gauss-Jordan ເທິງມາຕຣິກຂະຫຍາຍ [A | b]
    aug0 = np.hstack([A, b.reshape(-1, 1)])
    R, steps, pivots = gauss_jordan(aug0, n)
    inconsistent = any(
        all(R[i][j] == 0 for j in range(n)) and R[i][n] != 0 for i in range(n)
    )
    unique = (len(pivots) == n) and not inconsistent

    with st.expander("📖 ວິທີແກ້ດ້ວຍ Gauss-Jordan Elimination", expanded=open_all):
        st.markdown("**ຂັ້ນທີ 1:** ຂຽນລະບົບເປັນຮູບມາຕຣິກ  $A\\mathbf{x} = \\mathbf{b}$")
        st.latex(
            to_latex(A) + r"\begin{bmatrix}" + r" \\ ".join(names) + r"\end{bmatrix} = "
            + to_latex(b.reshape(-1, 1))
        )
        st.markdown(
            "**ຂັ້ນທີ 2:** ສ້າງມາຕຣິກຂະຫຍາຍ [ A | b ] ແລ້ວເຮັດ Row Reduction "
            "ຈົນດ້ານຊ້າຍເປັນມາຕຣິກຫົວໜ່ວຍ ໂດຍໃຊ້ 3 ຢ່າງ: ສະຫຼັບແຖວ, ຄູນ/ຫານແຖວດ້ວຍຄ່າຄົງທີ່, ເອົາແຖວໜຶ່ງບວກ/ລົບດ້ວຍຜົນຄູນຂອງອີກແຖວ"
        )
        show_steps(steps, n)
        st.markdown("**ຂັ້ນສຸດທ້າຍ:** ອ່ານຄຳຕອບຈາກຖັນຂວາສຸດ")
        if unique:
            for i in range(n):
                st.latex(rf"{names[i][0]}_{{{i + 1}}} = {fl(R[i][n])}")

    if inconsistent:
        st.error("ລະບົບນີ້ **ບໍ່ມີຄຳຕອບ** (ມີແຖວທີ່ສົມຜົນເປັນ 0 = ຄ່າທີ່ບໍ່ແມ່ນ 0, ເຊິ່ງເປັນໄປບໍ່ໄດ້)")
    elif not unique:
        st.error(
            f"rank(A) = {len(pivots)} < n = {n}: ລະບົບນີ້ **ມີຄຳຕອບຫຼາຍຢ່າງ** "
            f"(ມີຕົວແປອິດສະຫຼະ {n - len(pivots)} ຕົວ)"
        )
    else:
        x = np.array([float(R[i][n]) for i in range(n)])
        st.success("ຄຳຕອບ: " + ",  ".join(f"{names[i]} = {_fmt(x[i])}" for i in range(n)))
        st.latex(r"x = A^{-1}b = " + to_latex(x.reshape(-1, 1)))
        with st.expander("📖 ກວດສອບຄຳຕອບ (ແທນຄ່າກັບເຂົ້າສົມຜົນ)", expanded=open_all):
            for i in range(n):
                terms = " + ".join(f"({_fmt(A[i, j])})({_fmt(x[j])})" for j in range(n))
                st.latex(rf"\text{{ສົມຜົນທີ }} {i + 1}: {terms} = {_fmt(A[i] @ x)} \;\;(b_{{{i + 1}}} = {_fmt(b[i])})")
        st.caption(f"ກວດສອບ: ||Ax − b|| = {np.linalg.norm(A @ x - b):.2e}")

# ------------------------------------------------------------
# 4) Eigenvalues
# ------------------------------------------------------------
else:
    st.header("ຄ່າເຈາະຈົງ ແລະ ເວັກເຕີເຈາະຈົງ")
    st.latex(r"A\mathbf{v} = \lambda \mathbf{v}")
    n = st.slider("ຂະໜາດມາຕຣິກ n × n", 2, 4, 2)
    A = matrix_input("A", n, n, "eig", default="random")
    show("ມາຕຣິກ A", A)

    vals, vecs = np.linalg.eig(A)
    coefs = np.poly(A)  # ສຳປະສິດຂອງ det(λI − A)

    with st.expander("📖 ວິທີຫາຄ່າເຈາະຈົງ (Eigenvalues)", expanded=open_all):
        st.markdown(
            "**ຂັ້ນທີ 1: ຕັ້ງສົມຜົນ** — ຈາກ $A\\mathbf{v} = \\lambda\\mathbf{v}$ "
            "ຍ້າຍຂ້າງໄດ້ $(A - \\lambda I)\\mathbf{v} = \\mathbf{0}$. "
            "ເພື່ອໃຫ້ມີ $\\mathbf{v} \\neq \\mathbf{0}$ ຕ້ອງມີ det ເປັນ 0:"
        )
        st.latex(r"\det(A - \lambda I) = 0")

        st.markdown("**ຂັ້ນທີ 2: ສ້າງມາຕຣິກ** $A - \\lambda I$ (ລົບ λ ອອກຈາກຄ່າເທິງເສັ້ນຂວາງຫຼັກ)")
        sym_rows = [
            [
                (rf"{_fmt(A[i, j])} - \lambda" if i == j else _fmt(A[i, j]))
                for j in range(n)
            ]
            for i in range(n)
        ]
        st.latex(str_latex(sym_rows))

        st.markdown("**ຂັ້ນທີ 3: ຄິດໄລ່ det ເພື່ອໄດ້ພຫຸນາມລັກສະນະ (characteristic polynomial)**")
        if n == 2:
            tr, dt = np.trace(A), np.linalg.det(A)
            st.latex(
                rf"\det(A-\lambda I) = ({_fmt(A[0,0])}-\lambda)({_fmt(A[1,1])}-\lambda) "
                rf"- ({_fmt(A[0,1])})({_fmt(A[1,0])})"
            )
            st.latex(r"= \lambda^2 - \operatorname{tr}(A)\,\lambda + \det(A)")
            st.latex(rf"= \lambda^2 - ({_fmt(tr)})\lambda + ({_fmt(dt)}) = {poly_latex(coefs)}")
            disc = tr ** 2 - 4 * dt
            st.markdown("**ຂັ້ນທີ 4: ແກ້ສົມຜົນກຳລັງສອງ ດ້ວຍສູດ**")
            st.latex(r"\lambda = \frac{\operatorname{tr}(A) \pm \sqrt{\Delta}}{2}, \quad \Delta = \operatorname{tr}(A)^2 - 4\det(A)")
            st.latex(rf"\Delta = ({_fmt(tr)})^2 - 4({_fmt(dt)}) = {_fmt(disc)}")
            if disc > 1e-10:
                st.markdown("Δ > 0 → ມີຄ່າເຈາະຈົງເປັນຈຳນວນຈິງ 2 ຄ່າທີ່ຕ່າງກັນ")
            elif abs(disc) <= 1e-10:
                st.markdown("Δ = 0 → ມີຄ່າເຈາະຈົງຊ້ຳກັນ 1 ຄ່າ")
            else:
                st.markdown("Δ < 0 → ຄ່າເຈາະຈົງເປັນຈຳນວນຊັບຊ້ອນ (complex)")
        else:
            st.latex(rf"\det(A-\lambda I) = {poly_latex(coefs)} = 0")
            st.markdown(
                f"**ຂັ້ນທີ 4:** ຫາຮາກຂອງພຫຸນາມກຳລັງ {n} ນີ້ (ໃຊ້ວິທີຕົວເລກ numpy.roots)"
            )
        roots = np.roots(coefs)
        st.latex(r"\lambda = " + ",\\; ".join(_fmt(r) for r in roots))

    st.subheader("ຜົນລັບ")
    for i in range(n):
        st.markdown(f"**λ{i + 1} = {_fmt(vals[i])}**")
        st.latex(r"\mathbf{v}_{" + str(i + 1) + r"} = " + to_latex(vecs[:, i].reshape(-1, 1)))
        with st.expander(f"📖 ວິທີຫາ v{i + 1} ຂອງ λ{i + 1} = {_fmt(vals[i])}", expanded=open_all):
            st.markdown("**ຂັ້ນທີ 1:** ແທນ λ ເຂົ້າໃນ $A - \\lambda I$")
            st.latex(r"A - \lambda I = " + to_latex(A - vals[i] * np.eye(n)))
            st.markdown(
                "**ຂັ້ນທີ 2:** ແກ້ລະບົບ $(A - \\lambda I)\\mathbf{v} = \\mathbf{0}$ "
                "ເນື່ອງຈາກ det = 0 ແຖວຕ່າງໆຂຶ້ນຕໍ່ກັນ ຈຶ່ງມີຄຳຕອບທີ່ບໍ່ແມ່ນສູນຢ່າງໜ້ອຍ 1 ທິດທາງ"
            )
            st.markdown("**ຂັ້ນທີ 3:** ປັບຄວາມຍາວເວັກເຕີໃຫ້ເທົ່າ 1 (normalize)")
            st.latex(rf"\|\mathbf{{v}}_{{{i + 1}}}\| = {np.linalg.norm(vecs[:, i]):.4f}")
            st.markdown("**ຂັ້ນທີ 4: ກວດສອບ** $A\\mathbf{v} = \\lambda\\mathbf{v}$")
            st.latex(
                r"A\mathbf{v} = " + to_latex((A @ vecs[:, i]).reshape(-1, 1))
                + r", \quad \lambda\mathbf{v} = " + to_latex((vals[i] * vecs[:, i]).reshape(-1, 1))
            )

    st.caption(
        f"ກວດສອບ: ຜົນບວກຄ່າເຈາະຈົງ = {_fmt(np.sum(vals))} (= trace), "
        f"ຜົນຄູນ = {_fmt(np.prod(vals))} (= det)"
    )