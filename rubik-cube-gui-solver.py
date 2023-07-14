from direct.showbase.ShowBase import ShowBase
from direct.showbase.Loader import Loader
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import *
from panda3d.core import *
from direct.task import Task
from itertools import *
import functools
import kociemba
import math

#画面サイズ
GROUND_X = 1024
GROUND_Y = 1024
#パネルサイズ
PANEL_WIDTH = 30
#極座標表示
r = 10.0
theta = 30.0
phi = 10.0



FACE_COLOR_ID_BLACK  = 0 # キューブの見えない面に使われている
FACE_COLOR_ID_WHITE  = 1
FACE_COLOR_ID_RED    = 2
FACE_COLOR_ID_BLUE   = 3
FACE_COLOR_ID_ORANGE = 4
FACE_COLOR_ID_GREEN  = 5
FACE_COLOR_ID_YELLOW = 6
FACE_COLOR_ID_MAX    = FACE_COLOR_ID_YELLOW

# 色の ID と実際の色との対応表
FACE_COLORS = [
    (0.0, 0.0, 0.0, 1.0), # black as (R, G, B, A)
    (1.0, 1.0, 1.0, 1.0), # white
    (1.0, 0.0, 0.0, 1.0), # red
    (0.3, 0.3, 1.0, 1.0), # blue
    (1.0, 0.5, 0.0, 1.0), # orange
    (0.0, 0.9, 0.0, 1.0), # green
    (1.0, 1.0, 0.0, 1.0)] # yellow

FACE_COLOR_INITIAL = {
    '.' : FACE_COLOR_ID_BLACK,
    'W' : FACE_COLOR_ID_WHITE, 'R' : FACE_COLOR_ID_RED,
    'B' : FACE_COLOR_ID_BLUE,  'O' : FACE_COLOR_ID_ORANGE,
    'G' : FACE_COLOR_ID_GREEN, 'Y' : FACE_COLOR_ID_YELLOW, }

# RubikCubeInputPanels の初期配置
INITIAL_ARRANGEMENT = [[FACE_COLOR_INITIAL[c] for c in list(s)] for s in [
    "...WWW......",
    "...WWW......",
    "...WWW......",
    "GGGRRRBBBOOO",
    "GGGRRRBBBOOO",
    "GGGRRRBBBOOO",
    "...YYY......",
    "...YYY......",
    "...YYY......"]]

#
#
#
class Permutation:
    def __init__(self, table : list, m = None):
        self.table = table.copy()
        if m is None: m = max(self.table)
        n = len(self.table)
        if n < m: self.table.extend(range(n, m))
        self.check_validity()

    # 置換になっているか検証し, だめだったら例外を投げる
    # デバッグ時には大変お世話になっていますが, 完成後は要らないかも
    def check_validity(self):
        m = len(self.table)
        if any([not p[0] == p[1] for p in zip(sorted(self.table), range(m))]):
            raise(ValueError("Error: Invalid permutation"))

    # 置換の合成
    def __mul__(self, other):
        return Permutation([self.table[c] for c in other.table])

    # 逆置換
    def invert(self):
        m = len(self.table)
        t = [0 for _ in range(0, m)]
        for i in range(0, m): t[self.table[i]] = i
        return Permutation(t)

    # 互換の積に分解し, そのリストを返す
    # 互換は [i, j] (i < j) という形のリストで表される
    def transpositions(self):
        t = self.table.copy()
        s = []
        for i in range(0, len(t) - 1):
            j = t[i: ].index(i) + i
            if j == i: continue
            t[i], t[j] = t[j], t[i]
            s.append([i, j])
        return reversed(s)

    # 隣接互換 [i, i + 1] の積に分解し, そのリストを返す
    # テスト用：以下のコードを実行すると x と y は同じになる
    # x = Permutation([5, 3, 4, 1, 2, 0])
    # a = x.adjcent_transpositions()
    # y = functools.reduce(lambda a, b: a * b, map(lambda t: Permutation.cyclic(t, len(c.table)), a))
    def adjcent_transpositions(self):
        adj = []
        for i, j in self.transpositions():
            adj.extend([k - 1, k] for k in range(j, i, -1))
            adj.extend([k, k + 1] for k in range(i + 1, j))
        return adj

    # リストを受け取り, 置換を適用したものを返す
    def permute(self, x : list):
        m = len(self.table)
        y = x.copy()
        for i in range(0, m): x[self.table[i]] = y[i]

    # 置換を n 乗する (指数 n は負の整数でもよい)
    # 指数 n は高々一桁程度を想定しており, 速度などは何ら考慮されていない
    def power(self, n):
        if n <  0: return self.invert().power(-n)
        if n == 0: return self.cyclic([], len(self.table))
        if n >  0: return self * self.power(n - 1)

    # 二乗する
    def square(self): return self.power(2)

    # 巡回置換を生成する
    @staticmethod
    def cyclic(cycle, m = None):
        if m is None: m = max(cycle) + 1
        p = list(range(0, m))
        for i in range(0, len(cycle)):
            p[cycle[i - 1]] = cycle[i]
        return Permutation(p, m)

    # 恒等置換を生成する
    @staticmethod
    def identity(m):
        return Permutation.cyclic([], m)

# CubeSymmetry は立方体の回転対称性を4文字 0, 1, 2, 3 の置換として表現して取り扱う Permutation の派生クラスである
# 立方体は (0, 0, 0) を中心に xy, yz, zx 平面と平行な面を持っていると仮定する
# 置換における各文字 i は, 方向ベクトルを CUBE_DIAG_DIR_VEC[i] とする立方体の対角線を表す
class CubeSymmetry(Permutation):
    CUBE_DIAG_DIR_VEC = [(-1, -1, +1), (+1, -1, +1), (+1, +1, +1), (-1, +1, +1)]

    def __init__(self, table : list):
        super().__init__(table)

    # 以下は親クラスのメソッド
    # Python の流儀がわからないが, 親クラスで仮想関数にしておくほうが標準的だったかもしれない
    def __mul__(self, other):
        result = super().__mul__(other)
        result.__class__ = CubeSymmetry
        return result

    def power(self, n):
        result = super().power(n)
        result.__class__ = CubeSymmetry
        return result

    def invert(self):
        result = super().invert()
        result.__class__ = CubeSymmetry
        return result

    # このインスタンスの表す回転を四元数 panda3d.core.Quat として返す
    # 計算のロジックを間違えないようにかなり回りくどい計算を行っているが,
    # 入力は高々 4! = 24 通りなので, 速度が欲しいなら, あらかじめテーブルでも作っておくべきであろう
    # 実際は, 今のところ特に速度は必要としていないので, これでよしとしている
    def as_quaternion(self):
        # 以下の関数は隣接互換 [i, i + 1] と対応する回転を表す四元数を返す
        # 隣接互換 [i, i + 1] はベクトル v[i] + v[i + 1] を軸とする 180 度回転と対応している
        # ここで v[i] = CUBE_DIAG_DIR_VEC[i] である
        def adj_trans_to_quad(t):
            i = t[0]; assert t[1] == i + 1
            v = Vec3(CubeSymmetry.CUBE_DIAG_DIR_VEC[i])
            w = Vec3(CubeSymmetry.CUBE_DIAG_DIR_VEC[i + 1])
            q = Quat(); q.setFromAxisAngle(180, (v + w).normalized()); return q
        # 隣接互換の積に分解した後, 上の関数で変換したものを全部かける
        return functools.reduce(lambda p, q: p * q,
            map(adj_trans_to_quad, self.adjcent_transpositions()),
            Quat(1, 0, 0, 0))

    # このインスタンスの表す回転を HPR で返す
    def as_hpr(self):
        # 四元数を HPR に変換するルーチンはライブラリに用意されている
        h, p, r = self.as_quaternion().getHpr()
        # しかし角度の計算誤差がひどい場合がある
        # 正しい角度は 90 度の倍数に限られるから, それに丸めてしまった
        def round90(x): return round(x / 90.0) * 90
        return Vec3(round90(h), round90(p), round90(r))

    # as_face_permutation は, このインスタンスの表す回転を, 立方体の面の置換として返す
    # これも as_quaternion と同様にまわりくどいことをしているが, あらかじめテーブルでも作っておくほうがよかろう
    # まず, 隣接互換 [i, i + 1] が引き起こす面の入れ替えをリストとして用意しておく
    FACE_PERMUTATION_BY_ADJCENT_TRANSPOSITION_0 = [
        [["U", "F"], ["L", "R"], ["B", "D"]], # (i = 0) U <-> F, L <-> R, B <-> D
        [["U", "R"], ["L", "D"], ["F", "B"]], # (i = 1)
        [["U", "B"], ["L", "R"], ["F", "D"]]] # (i = 2)
    # kociemba library の面の番号付け URFDLB に応じて, { 0, 1, ..., 6 } の置換に変換しておく
    def _convert_face_permutation(t):
        def convert(pair):
            i, j = map(lambda f: list("URFDLB").index(f), pair)
            return Permutation.cyclic([i, j], 6)
        return functools.reduce(lambda p, q: p * q, map(convert, t))
    FACE_PERMUTATION_BY_ADJCENT_TRANSPOSITION \
        = list(map(_convert_face_permutation, FACE_PERMUTATION_BY_ADJCENT_TRANSPOSITION_0))
    def as_face_permutation(self):
        def adj_trans_to_face_permutation(t):
            i = t[0]; assert t[1] == i + 1
            return self.FACE_PERMUTATION_BY_ADJCENT_TRANSPOSITION[i]
        return functools.reduce(lambda p, q: p * q,
            map(adj_trans_to_face_permutation, self.adjcent_transpositions()),
            Permutation.identity(6))

# キューブの場所には 0 から 26 までの番号がつけられている
# 以下は面 U, R, F, D, L, B に属するキューブの番号を一覧にしたもの
FACE_CUBE_INDICES = {
    'U': [  0,  1,  2,  3,  4,  5,  6,  7,  8],
    'R': [  8,  5,  2, 17, 14, 11, 26, 23, 20],
    'F': [  6,  7,  8, 15, 16, 17, 24, 25, 26],
    'D': [ 24, 25, 26, 21, 22, 23, 18, 19, 20],
    'L': [  0,  3,  6,  9, 12, 15, 18, 21, 24],
    'B': [  2,  1,  0, 11, 10,  9, 20, 19, 18]
}

# 各面における90度回転 (ルービックキューブの外側からみて時計回り) がキューブの配列に引き起こす置換
FACE_CUBE_PERMUTATIONS = {
    'U': Permutation.cyclic([ 0,  1,  2,  5,  8,  7,  6,  3], 27).square(),
    'R': Permutation.cyclic([ 8,  5,  2, 11, 20, 23, 26, 17], 27).square(),
    'F': Permutation.cyclic([ 6,  7,  8, 17, 26, 25, 24, 15], 27).square(),
    'D': Permutation.cyclic([24, 25, 26, 23, 20, 19, 18, 21], 27).square(),
    'L': Permutation.cyclic([ 0,  3,  6, 15, 24, 21, 18,  9], 27).square(),
    'B': Permutation.cyclic([ 2,  1,  0,  9, 18, 19, 20, 11], 27).square()
}

# 立方体の対称性 (x, y, z 軸まわり, 右ねじの向きに 90 度回転)
CUBE_SYMMETRY_X_90DEG = CubeSymmetry([2, 3, 1, 0])  # = cyclic(0, 2, 1, 3)
CUBE_SYMMETRY_Y_90DEG = CubeSymmetry([1, 3, 0, 2])  # = cyclic(0, 1, 3, 2)
CUBE_SYMMETRY_Z_90DEG = CubeSymmetry([1, 2, 3, 0])  # = cyclic(0, 1, 2, 3)

# 回転の引き起こす立方体の対称性
FACE_CUBE_SYMMETRIES = {
    'U': CUBE_SYMMETRY_Z_90DEG.invert(),
    'R': CUBE_SYMMETRY_X_90DEG.invert(),
    'F': CUBE_SYMMETRY_Y_90DEG,
    'D': CUBE_SYMMETRY_Z_90DEG,
    'L': CUBE_SYMMETRY_X_90DEG,
    'B': CUBE_SYMMETRY_Y_90DEG.invert(),
}

# 対応する置換を HPR で表したもの
FACE_CUBE_ROTATIONS_HPR = {
    'U': Vec3(-90,   0,   0),
    'R': Vec3(  0, -90,   0),
    'F': Vec3(  0,   0, +90),
    'D': Vec3(+90,   0,   0),
    'L': Vec3(  0, +90,   0),
    'B': Vec3(  0,   0, -90)
}

class RubikCubeModel():
    def __init__(self, showbase):
        self.showbase = showbase
        self.root = showbase.render.attachNewNode(PandaNode('rotating_cubes_node'))
        self.node_cubes_all = showbase.render.attachNewNode(PandaNode('rotating_cubes_node'))
        self.node_cubes_rot = showbase.render.attachNewNode(PandaNode('rotating_cubes_node'))
        self.node_cubes_all.reparentTo(self.root)
        self.node_cubes_rot.reparentTo(self.root)
        self.callback = None
        self.rotation_interval = 1
        # 27個のキューブを生成する
        self.cubes = [None for _ in range(27)]
        for i, j, k in product(range(3), range(3), range(3)):
            c = self.Cube(showbase)
            c.root.reparentTo(self.node_cubes_all)
            self.cubes[i * 9 + j * 3 + k] = c
        # 位置を調整する
        self.justify_cube_arrangements()
        # とりあえずデフォルトの色を塗っておく
        self.set_color()

    # 各キューブを表すクラス
    # キューブ全体の回転・移動は, self.root に対して操作をすればよい
    class Cube:
        def __init__(self, showbase):
            self.showbase = showbase
            self.root = self.showbase.render.attachNewNode(PandaNode('cube'))
            self.root.reparentTo(self.showbase.render)
            self.root.setPos(0, 0, 0)
            self.root.setHpr(0, 0, 0)
            # 以下のリストで示される位置と角度にしたがって面を配置し, 結果として立方体が作られる
            # 各面は self.root の子ノードとなっており, self.root.children[?] でアクセスできる
            # ToDo: 現在は face.egg というファイルを読み込んでいるが, このくらい外部ファイルとせずに, プログラムの中で生成できないのか？
            configure = [
                [Vec3( 0.0,  0.0,  0.5), Vec3(0,   0,   0)], # U
                [Vec3( 0.5,  0.0,  0.0), Vec3(0,   0,  90)], # R
                [Vec3( 0.0, -0.5,  0.0), Vec3(0,  90,   0)], # F
                [Vec3( 0.0,  0.0, -0.5), Vec3(0,   0, 180)], # D
                [Vec3(-0.5,  0.0,  0.0), Vec3(0,   0, -90)], # L
                [Vec3( 0.0,  0.5,  0.0), Vec3(0, -90,   0)]] # B
            for pos, hpr in configure:
                face = self._create_square()
                face.reparentTo(self.root)
                face.setColor(FACE_COLORS[FACE_COLOR_ID_BLACK])
                face.setHpr(hpr)
                face.setPos(pos)
            # キューブの回転と対応する置換
            self.rotation = CubeSymmetry([0,1,2,3])
            # キューブの面の色
            self.face_colors = [FACE_COLOR_ID_BLACK for _ in range(6)]

        # 点 (0, 0, 0) を中心とする辺の長さが 1 の正方形を作る
        @staticmethod
        def _create_square():
            format = GeomVertexFormat.get_v3n3()
            vdata = GeomVertexData("square vertices", format, Geom.UH_static)
            vertex = GeomVertexWriter(vdata, "vertex")
            normal = GeomVertexWriter(vdata, "normal")
            vertex.add_data3(+.5, +.5, 0)
            vertex.add_data3(-.5, +.5, 0)
            vertex.add_data3(-.5, -.5, 0)
            vertex.add_data3(+.5, -.5, 0)
            for _ in range(4): normal.add_data3(0, 0, 1)
            prim = GeomTriangles(Geom.UH_static)
            prim.add_vertices(0, 1, 2)
            prim.add_vertices(2, 3, 0)
            geom = Geom(vdata)
            geom.add_primitive(prim)
            node = GeomNode("square node")
            node.add_geom(geom)
            return NodePath(node)

        # キューブの面は self.root.children[?] でアクセスできるが,
        # キューブが回転した後は, どの面が上になっているか計算しないとわからない.
        # 以下のルーチンは self.rotation の情報をもとに面 U, R, F, D, L, B に対応する番号を返す
        def get_face_index(self, face):
            n = list("URFDLB").index(face)
            p = self.rotation.as_face_permutation().invert()
            return p.table[n]

        # 面 U, R, F, D, L, B を返す
        def get_face(self, face):
            return self.root.children[self.get_face_index(face)]

        # 指定した面の色を取得あるいは設定する
        def get_face_color(self, face):
            return self.face_colors[self.get_face_index(face)]
        def set_face_color(self, face, color_id):
            n = self.get_face_index(face)
            self.face_colors[n] = color_id
            self.root.children[n].setColor(FACE_COLORS[color_id])

    # ルービックキューブ全体を隠す, あるいは現す
    def hide(self): self.root.hide()
    def show(self): self.root.show()

    # ルービックキューブ全体の色を設定する
    # 色として None を指定するとデフォルトの色に設定される
    DEFAULT_COLORS = [FACE_COLOR_INITIAL[c] for c in list("WWWWWWWWWBBBBBBBBBRRRRRRRRRYYYYYYYYYGGGGGGGGGOOOOOOOOO")]
    def set_color(self, colors = None):
        if colors is None: colors = self.DEFAULT_COLORS
        for face, n in zip(list("URFDLB"), range(0, 6)):
            for i, j in zip(FACE_CUBE_INDICES[face], range(0, 9)):
                self.cubes[i].set_face_color(face, colors[9 * n + j])

    # ルービックキューブの色を取得する
    def get_color(self):
        c = []
        for face, n in zip(list("URFDLB"), range(0, 6)):
            for i, j in zip(FACE_CUBE_INDICES[face], range(0, 9)):
                c.append(self.cubes[i].get_face_color(face))
        return c

    # 計算誤差などのためにキューブの位置がずれていく可能性があるので,
    # キューブの配置などの情報から真の位置に設定しなおすルーチン
    # 実際のところ, あまりズレているようには見えないので, 必要ないかもしれない
    def justify_cube_arrangements(self):
        for i, j, k in product(range(3), range(3), range(3)):
            c = self.cubes[i * 9 + j * 3 + k]
            # キューブの位置を合わせる
            c.root.setPos((k - 1) * 1.05, (1 - j) * 1.05, (1 - i) * 1.05)
            # キューブの回転角を再設定する
            # なんで invert が必要なのかわからない. とてもやばい.
            c.root.setHpr(c.rotation.invert().as_hpr())

    # 面 f をキューブの外から見たときに時計回りに90度の回転を, 面 f に対して連続して k 回行う
    # 排他処理などは特にしていないので, 回転中にこのメソッドが呼び出されるとルービックキューブが壊れる
    def rotate_face(self, f, k):
        # k % 4 == 0 なら回転しないので帰る
        if k % 4 == 0: return
        # まず self.node_cubes_rot の回転角をゼロにしておく
        # しかる後に, f で指定される面にあるキューブたちを self.node_cubes_rot の子ノードにする
        # キューブの回転は, 実際には self.node_cubes_rot を回転させることで実行される
        self.node_cubes_rot.setHpr(0,0,0)
        for n in FACE_CUBE_INDICES[f]:
            self.cubes[n].root.wrtReparentTo(self.node_cubes_rot)
        # キューブの回転のアニメーションをするタスクを追加
        self.showbase.taskMgr.add(self.task_rotate_cubes, appendTask = True, extraArgs = [[
            # この回転によって動くことになるキューブのリスト
            FACE_CUBE_INDICES[f],
            # 最終的にキューブたちに対して引き起こされる置換
            FACE_CUBE_PERMUTATIONS[f].power(k),
            # キューブに引き起こされる回転
            FACE_CUBE_SYMMETRIES[f].power(k),
            # キューブに引き起こされる回転
            FACE_CUBE_ROTATIONS_HPR[f] * float(k),
            # 回転にかける時間 (in seconds)
            self.rotation_interval * abs(k)
        ]])

    # キューブが回転するアニメーションにおいて使われるコールバック関数
    # extra_args については, 上で taskMgr に task_rotate_cubes を追加しているところの説明を見よ
    def task_rotate_cubes(self, extra_args, task):
        # 指定された時間内は, キューブを回転させるというタスクを継続する
        hpr = extra_args[3]
        if task.time < extra_args[4]:
            r = task.time / extra_args[4]
            self.node_cubes_rot.setHpr(hpr * r)
            return Task.cont
        # 以下はタスク終了後の処理
        self.node_cubes_rot.setHpr(hpr)
        # 回転されるキューブは self.node_cubes_rot が親ノードになっているので, これを元に戻す
        for n in extra_args[0]:
            self.cubes[n].root.wrtReparentTo(self.node_cubes_all)
        # キューブのリストを置換し, 計算誤差の累積を避けるためにキューブの位置を再調整する
        for n in extra_args[0]:
            self.cubes[n].rotation = extra_args[2] * self.cubes[n].rotation
        extra_args[1].permute(self.cubes)
        self.justify_cube_arrangements()
        # 回転が終わったよって伝える
        if self.callback is not None: self.callback()



        return Task.done

# ルービックキューブの解法を保存するためのクラス
# __init__ の処理を隠したかったのと, 双方向イテレータとして扱いたかっただけで, たいしたことはしていない
class RubikCubeSolution:
    # kociemba library の出力を整形しておく
    # 例えば "F' B D2" は [["F", -1], ["B", 1], ["D", 2]] に変換されて self.moves に保存される
    def __init__(self, solution_str):
        self.current = 0
        self.moves = []
        for s in solution_str.split(" "):
            if len(s) == 1:
                self.moves.append([s, 1])
            else:
                f = s[0]
                d = (-1 if s[1] == "'" else int(s[1]))
                self.moves.append([f, d])

    # 次の手順を返す
    def next(self):
        if not self.current < len(self.moves): raise StopIteration()
        result = self.moves[self.current]
        self.current += 1
        return result

    # 前の手順を返す
    def prev(self):
        if not self.current > 0: raise StopIteration()
        self.current -= 1
        return self.moves[self.current]

#
#
#
class RubikCubeSolverGUI(ShowBase):
    global r, theta, phi
    def __init__(self):
        ShowBase.__init__(self)
        # 全体的な設定
        self.disableMouse()
        # 3Dモデルのカメラ設定
        self.r = r
        self.theta = theta
        self.phi = phi
        self.x = self.r*math.sin(self.theta)*math.cos(self.phi)
        self.y = self.r*math.sin(self.theta)*math.sin(self.phi)
        self.z = self.r*math.cos(self.theta)
        self.camera.setPos(self.x, self.y, self.z)
        self.camera.lookAt(0, 0, 0)
        self.setBackgroundColor(0, 0, 0)
        # ルービックキューブ入力用画面
        self.input_screen = RubikCubeSolverGUI_InputScreen(self)
        self.input_screen.commands["solve"] = self.start_solving
        # サンプル群
        self.input_screen.commands["sample1"] = self.sample1
        self.input_screen.commands["sample2"] = self.sample2
        self.input_screen.commands["sample3"] = self.sample3
        # 終了コマンド
        self.input_screen.commands["quit"] = self.quit
        # ヘルプ画面
        #self.input_screen.commands["help"] = self.help
        # ルービックキューブのアニメーション用画面
        self.animation_screen = RubikCubeSolverGUI_AnimationScreen(self)
        self.animation_screen.commands["back"] = self.quit_animation_screen
        self.animation_screen.commands["move end"] = self.callback_move_end

        self.accept("q", self.q_key);
        self.accept("w", self.w_key);
        self.accept("a", self.a_key);
        self.accept("s", self.s_key);
        self.accept("z", self.z_key);
        self.accept("x", self.x_key);
        
    
    def q_key(self):
       self.r  = self.r + 5
       self.x = self.r*math.sin(self.theta)*math.cos(self.phi)
       self.y = self.r*math.sin(self.theta)*math.sin(self.phi)
       self.z = self.r*math.cos(self.theta)
       self.camera.setPos(self.x, self.y, self.z)
       self.camera.lookAt(0, 0, 0)

    def w_key(self):
       self.r = min(3, self.r - 5)
       self.x = self.r*math.sin(self.theta)*math.cos(self.phi)
       self.y = self.r*math.sin(self.theta)*math.sin(self.phi)
       self.z = self.r*math.cos(self.theta)
       self.camera.setPos(self.x, self.y, self.z)
       self.camera.lookAt(0, 0, 0)

    def a_key(self):
       self.theta = (self.theta+9)%360
       self.x = self.r*math.sin(self.theta)*math.cos(self.phi)
       self.y = self.r*math.sin(self.theta)*math.sin(self.phi)
       self.z = self.r*math.cos(self.theta)
       self.camera.setPos(self.x, self.y, self.z)
       self.camera.lookAt(0, 0, 0)

    def s_key(self):
       self.theta = (self.theta-9+360)%360
       self.x = self.r*math.sin(self.theta)*math.cos(self.phi)
       self.y = self.r*math.sin(self.theta)*math.sin(self.phi)
       self.z = self.r*math.cos(self.theta)
       self.camera.setPos(self.x, self.y, self.z)
       self.camera.lookAt(0, 0, 0)

    def z_key(self):
       self.phi = (self.phi+90)%360
       self.x = self.r*math.sin(self.theta)*math.cos(self.phi)
       self.y = self.r*math.sin(self.theta)*math.sin(self.phi)
       self.z = self.r*math.cos(self.theta)
       self.camera.setPos(self.x, self.y, self.z)
       self.camera.lookAt(0, 0, 0)

    def x_key(self):
       self.phi = (self.phi-90+360)%360
       self.x = self.r*math.sin(self.theta)*math.cos(self.phi)
       self.y = self.r*math.sin(self.theta)*math.sin(self.phi)
       self.z = self.r*math.cos(self.theta)
       self.camera.setPos(self.x, self.y, self.z)
       self.camera.lookAt(0, 0, 0)

    def q_key(self):
       self.r += 10
       self.x = self.r*math.sin(self.theta)*math.cos(self.phi)
       self.y = self.r*math.sin(self.theta)*math.sin(self.phi)
       self.z = self.r*math.cos(self.theta)
       self.camera.setPos(self.x, self.y, self.z)
       self.camera.lookAt(0, 0, 0)





    def start_solving(self):
        # kociemba library にパネルの配置を渡し, 解いてもらう (解けない場合は例外が発生する)
        try:
            c = self.input_screen.get_color()
            # kociemba library への入力形式に変換する
            table = {}
            for f, n in zip(list("URFDLB"), range(6)): table[c[9 * n + 4]] = f
            k = "".join([table[color] for color in c])
            # kociemba library に渡して解を得る
            print("configuration  :", "".join([str(x) for x in c]))
            print("kociemba input :", k)
            solution_str = kociemba.solve(k)
            print("kociemba output:", solution_str)
        except KeyError as e:
            print("KeyError", e)
            return None
        except ValueError as e:
            print(e)
            return None

        # アニメーション用画面に解を渡し, 画面の見た目を遷移させる
        self.animation_screen.solution = RubikCubeSolution(solution_str)
        self.input_screen.minimize()
        self.animation_screen.show()
        self.animation_screen.set_color(self.input_screen.get_color())
    def sample1(self):
        sample_arrangement = [int(c) for c in list("646361626161214151262321252525351545363432353464143454")]
        self.input_screen.set_color(sample_arrangement)
        
    def sample2(self):
        sample_arrangement = [int(c) for c in list("265665555111441341444433431314311333526226666256255222")]
        self.input_screen.set_color(sample_arrangement)

    def sample3(self):
        sample_arrangement = [int(c) for c in list("666666661644444444114111111333333233555555355222222225")]
        self.input_screen.set_color(sample_arrangement)

    def quit(self):
        self.userExit()


    def quit_animation_screen(self):
        self.animation_screen.hide()
        self.input_screen.maximize()
        self.input_screen.set_color(self.animation_screen.get_color())
        pass

    def callback_move_end(self):
        # パネルの色の再設定
        c = self.animation_screen.get_color()
        print("".join(map(lambda n: str(n), c)))
        self.input_screen.set_color(c)

#
# 入力用画面
#
class RubikCubeSolverGUI_InputScreen:
    def __init__(self, showbase):
        self.panels = InputPanels(showbase.pixel2d)
        self.panels.setPos(50, 0, -50)
        self.panels.set_panel_width(PANEL_WIDTH)
        # ボタンが押されたときに呼び出される関数を入れておくもの
        self.commands = {}
        # ボタンなどは, すべての以下のノードの子として作られる
        self.gui_root = showbase.pixel2d.attachNewNode(PandaNode(""))
        # ボタンとか作る
        BUTTONS = [
            [   0,  100, "Solve",    "solve",   None],
            [ 800,  100, "Help",     "help",    None],
            [1600,  100, "Quit",     "quit",    None],
            [   0,  400, "Sample 1", "sample1", None],
            [ 800,  400, "Sample 2", "sample2", None],
            [1600,  400, "Sample 3", "sample3", None]]
        for x, y, title, cmd, callback in BUTTONS:
            # パラメータの与え方は適当なので, あとできちんとする
            b = DirectButton(text = title, scale=0.075,
                             pos = (x/GROUND_X, 0, -y/GROUND_Y),  frameSize = (-3, 3, -1, 1),
                             command=self.on_button_click,text_pos = (0, -0.2),
                             extraArgs = [cmd])
            b.wrtReparentTo(self.gui_root)
            self.commands[cmd] = callback


    def on_button_click(self, cmd):
        callback = self.commands[cmd]
        if callback is not None: callback()
        else: print("No callback function is set. command =", cmd)

    # 色を取得/設定する
    def get_color(self):
        return self.panels.get_panels_colors()
    def set_color(self, colors):
        return self.panels.set_panels_colors(colors)
    # 表示・非表示・最小化など
    def show(self):
        self.panels.set_panel_width(PANEL_WIDTH)
        self.panels.show()
        self.gui_root.show()
    def hide(self):
        self.panels.hide()
        self.gui_root.hide()
    def minimize(self):
        self.panels.set_panel_width(PANEL_WIDTH)
        self.panels.disable()
        self.gui_root.hide()
        pass
    def maximize(self):
        self.panels.set_panel_width(PANEL_WIDTH)
        self.panels.enable()
        self.gui_root.show()

class InputPanels(DirectFrame):
    # 各面の中心の (行, 列) 番号を面 U, R, F, D, L, B の順に
    FACE_CENTERS = [[1, 4, 'U'], [4, 7, 'R'], [4, 4, 'F'], [7, 4, 'D'], [4, 1, 'L'], [4, 10, 'B']]

    def __init__(self, parent=None, **kw):
        # Merge keyword options with default options, and initialize superclasses
        self.defineoptions(kw, ())
        DirectFrame.__init__(self, parent)
        # パネルの幅 (in pixels)
        self.panel_width = PANEL_WIDTH
        # 初期配置をもとにして, 入力用のパネルを作ります
        self.panel_holder = [[None for _ in range(12)] for _ in range(9)]
        for i, j in product(range(9), range(12)):
            if INITIAL_ARRANGEMENT[i][j] == FACE_COLOR_ID_BLACK: continue
            self.panel_holder[i][j] = self._create_panel(i, j, INITIAL_ARRANGEMENT[i][j])
        # panel_holder にはパネルがない場所もあるので, パネル全体をリストとしてまとめておく
        # ここでは kociemba library の入力の順序でパネルを並べておくことにした
        self.panels = []
        for c in self.FACE_CENTERS:
            p = range(c[0] - 1, c[0] + 2); q = range(c[1] - 1, c[1] + 2)
            self.panels.extend([self.panel_holder[i][j] for i, j in product(p, q)])
        # 初期カーソル位置
        self.cursor = (0, 3)
        # control が押されている状態かどうか判定するためのもの
        # Keyboard.IsKeyDown のようなものが見つからなかったので……
        self.control_down = False
        self.accept("control",     self.on_control_down)
        self.accept("control-up",  self.on_control_up)
        # キーボード操作
        self.accept("arrow_up",    self.move_cursor_u)
        self.accept("arrow_right", self.move_cursor_r)
        self.accept("arrow_down",  self.move_cursor_d)
        self.accept("arrow_left",  self.move_cursor_l)
        # 色名に対応するキー入力により, 選択中のパネルの色を変更する
        for c in ['w', 'r', 'b', 'o', 'g', 'y']:
            self.accept(c, self.change_color, [[c]])

    # 入力用のパネルを作る
    def _create_panel(self, i, j, c):
        p = self.Panel(self)
        p.setPos(j * self.panel_width, 0, -i * self.panel_width)
        p.set_panel_color(c)
        p.fore.bind(DGG.B1RELEASE, self.panel_mouse_click, [[p, i, j]])
        p.fore.bind(DGG.ENTER, self.panel_mouse_enter, [[p, i, j]])
        p.fore.bind(DGG.EXIT, self.panel_mouse_leave, [[p, i, j]])
        return p

    # パネルのサイズを変更する
    def get_panel_width(self): return self.panel_width
    def set_panel_width(self, width):
        self.panel_width = width
        for i, j in product(range(9), range(12)):
            p = self.panel_holder[i][j]
            if p is None: continue
            p.set_panel_width(width)
            p.setPos(j * width, 0, -i * width)

    # パネルがクリックされたとき
    def panel_mouse_click(self, extra_args, mouse_pos):
        panel = extra_args[0]
        self.cursor = (extra_args[1], extra_args[2])
        if self.control_down:
            panel.select()
        else:
            # パネルが複数枚選択されている場合は選択を解除する
            # そうでない場合はクリックされたパネルの色を変える
            selected_panels = self.get_selected_panels()
            for p in selected_panels: p.unselect()
            panel.select()
            if len(selected_panels) <= 1: panel.rotate_color()

    def get_selected_panels(self):
        return list(filter(lambda p: p.selected, self.panels))

    # パネルの全選択を解除する
    def unselect_all(self):
        for p in self.get_selected_panels(): p.unselect()

    # マウスがパネルに入ったとき
    def panel_mouse_enter(self, extra_args, mouse_pos):
        extra_args[0].on_mouse_enter()
        return

    # マウスがパネルから出たとき
    def panel_mouse_leave(self, extra_args, mouse_pos):
        extra_args[0].on_mouse_leave()
        return

    # カーソルの上下左右移動
    def move_cursor_u(self): self.move_cursor(self.cursor[0] - 1, self.cursor[1])
    def move_cursor_d(self): self.move_cursor(self.cursor[0] + 1, self.cursor[1])
    def move_cursor_r(self): self.move_cursor(self.cursor[0], self.cursor[1] + 1)
    def move_cursor_l(self): self.move_cursor(self.cursor[0], self.cursor[1] - 1)
    def move_cursor(self, row, col):
        if row < 0: row = 0
        if row >= len(self.panel_holder): row = len(self.panel_holder) - 1
        if col < 0: col = 0
        if col >= len(self.panel_holder[0]): col = len(self.panel_holder[0]) - 1
        p = self.panel_holder[row][col]
        if p is not None:
            self.unselect_all()
            self.cursor = (row, col)
            p.select()

    # 選択されているものの色を変更する
    def change_color(self, extra_args):
        color = FACE_COLOR_INITIAL[extra_args[0].upper()]
        for p in self.get_selected_panels(): p.set_panel_color(color)

    # すべてのパネルの色をリストとして取得する
    def get_panels_colors(self):
        return [p.color_id for p in self.panels]

    # すべてのパネルに対して色を設定する
    def set_panels_colors(self, color):
        for p, c in zip(self.panels, color):
            p.set_panel_color(c)

    # 色の設定ができるように / できないようにする
    def enable(self):
        pass
    def disable(self):
        pass

    # control キーが押されているかどうかの確認用
    def on_control_down(self): self.control_down = True
    def on_control_up(self):   self.control_down = False

    # 入力用のパネルを実装するクラス
    class Panel(DirectFrame):
        def __init__(self, parent, **kw):
            self.defineoptions(kw, ())
            DirectFrame.__init__(self, parent)
            # メンバ変数の設定
            self.back = DirectFrame(state=DGG.NORMAL, parent=self)
            self.fore = DirectFrame(state=DGG.NORMAL, parent=self)
            self.color_id = FACE_COLOR_ID_BLACK
            self.selected = False
            self.set_panel_width(parent.panel_width)
            self.fore["frameColor"] = FACE_COLORS[FACE_COLOR_ID_WHITE]
            self.unselect()

        # パネルの色を取得あるいは設定する
        def get_panel_color(self):
            return self.color_id
        def set_panel_color(self, color_id):
            self.color_id = color_id
            self.fore["frameColor"] = FACE_COLORS[color_id]

        # パネルの幅を設定する
        def set_panel_width(self, w):
            b = w * 0.03
            self.back["frameSize"] = (0, w, -w, 0)
            self.fore["frameSize"] = (b, w - b, - w + b, -b)

        # 色をローテーションする
        def rotate_color(self):
            if self.color_id < FACE_COLOR_ID_MAX:
                self.set_panel_color(self.color_id + 1)
            else:
                self.set_panel_color(1)

        # パネルを選択する
        # とりあえずパネルの縁の色を変えているが, もっとかっこよくしたい
        def select(self):
            self.selected = True
            self.back["frameColor"] = (1.0, 0.5, 0.5, 1.0)

        # パネルを選択されていない状態にする
        def unselect(self):
            self.selected = False
            self.back["frameColor"] = (0.0, 0.0, 0.0, 1.0)

        # マウスがパネルの上に侵入したときに呼ばれる
        # かっこいい色にする処理を書いてください
        def on_mouse_enter(self): pass

        # マウスがパネルから出て行ったときに呼ばれる
        # かっこいい色から元に戻しておいてください
        def on_mouse_leave(self): pass

class RubikCubeSolverGUI_AnimationScreen:
    def __init__(self, showbase):
        # 3D 表示されたルービックキューブ
        self.cube_model = RubikCubeModel(showbase)
        self.cube_model.callback = self.rubik_cube_callback
        self.cube_model.hide()
        # ルービックキューブの解
        self.solution = None
        # アニメーションの継続 (+1:順方向へ継続, 0:停止, -1:逆方向)
        self.continue_animation = 0
        # ボタンが押されたときに呼び出される関数を入れておくもの
        self.commands = {}
        # ボタンの一覧
        self.buttons = {}
        # GUI の部品は, すべての以下のノードの子として作られる
        self.gui_root = showbase.pixel2d.attachNewNode(PandaNode(""))
        self.gui_root.hide()
        # ボタンとか作る
        BUTTONS = [
            [1900, 80, "<<",   "prev cont",  self.command_prev_cont],
            [2200, 80, "<",    "prev",       self.command_prev],
            [2500, 80, "stop", "stop",       self.command_stop],
            [2800, 80, ">",    "next",       self.command_next],
            [3100, 80, ">>",   "next cont",  self.command_next_cont],
            [3100, 80, ">>",   "next cont",  self.command_next_cont],
            [3400, 80, "back", "back",       None]]
        for x, y, title, cmd, callback in BUTTONS:
            # パラメータの与え方は適当なので, あとできちんとする
            b = DirectButton(text = title, scale=0.07,
                             pos = (x/GROUND_X, 0, -y/GROUND_Y),
                             frameSize=(-2,2,-1,1),text_pos = (0, -0.2),
                             command=self.on_button_click,
                             extraArgs = [cmd])
            self.buttons[cmd] = b
            b.wrtReparentTo(self.gui_root)
            self.commands[cmd] = callback
        self.mySound = showbase.loader.loadSfx("levelup.ogg")


    def on_button_click(self, cmd):
        callback = self.commands[cmd]
        if callback is not None: callback()
        else: print("No callback function is set. command =", cmd)

    def command_next(self):
        try:
            m = self.solution.next()
            self.disable_buttons()
            self.cube_model.rotate_face(m[0], m[1])
        except StopIteration:
            self.continue_animation = 0
            self.enable_buttons()
            pass

    def command_prev(self):
        try:
            m = self.solution.prev()
            self.disable_buttons()
            self.cube_model.rotate_face(m[0], -m[1])
        except StopIteration:
            self.continue_animation = 0
            self.enable_buttons()

    def command_next_cont(self):
        self.continue_animation = +1
        self.command_next()
        pass

    def command_prev_cont(self):
        self.continue_animation = -1
        self.command_prev()
        pass

    def command_stop(self):
        self.continue_animation = 0
        pass



    BUTTON_NAMES = ["next", "next cont", "prev", "prev cont"]
    def disable_buttons(self):
        for c in self.BUTTON_NAMES:
            self.buttons[c]["state"] = DGG.DISABLED

    def enable_buttons(self):
        for c in self.BUTTON_NAMES:
            self.buttons[c]["state"] = DGG.NORMAL

    # 色の取得・設定
    def get_color(self):
        return self.cube_model.get_color()
    def set_color(self, colors):
        self.cube_model.set_color(colors)

    # ルービックキューブ回転処理が終了したときに呼び出される
    # 現在の GUI の状況に応じて, 次の処理を行う
    def rubik_cube_callback(self):
        if "move end" in self.commands:
            (self.commands["move end"])()
            self.mySound.play()
        if self.continue_animation == +1:
            self.command_next()
        elif self.continue_animation == -1:
            self.command_prev()
        else:
            self.enable_buttons()

    def show(self):
        self.cube_model.show()
        self.gui_root.show()

    def hide(self):
        self.cube_model.hide()
        self.gui_root.hide()

#loadPrcFileData('', 'win-size 1024 1024')
app = RubikCubeSolverGUI()
sample_arrangement = [int(c) for c in list("646455232644525515565261221154233124261343363313614614")]
app.input_screen.set_color(sample_arrangement)
app.run()
