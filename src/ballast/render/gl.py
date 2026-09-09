"""`moderngl`-based rasterizer: real-time rendering with zero CUDA
dependency, chosen specifically because it goes through OpenGL rather than
CUDA and can therefore run on a machine's integrated GPU, completely
orthogonal to VRAM contention on the discrete GPU (verified at build time:
`moderngl.create_context(standalone=True)` bound the Intel iGPU on this
machine, not the RTX card). No compiled CUDA extension, no MSVC toolchain,
no dependency-version wheel gap on an sm_120 card -- the exact failure
modes a PyTorch3D/nvdiffrast route would hit on this hardware.

Produces an RGB image and a silhouette mask (alpha channel) per pose. The
mask is cross-checked against the pure-NumPy `cpu_fallback` rasterizer in a
golden-image test: two independent implementations of the same projection
agreeing is real evidence neither has a sign or winding bug.
"""

from __future__ import annotations

import numpy as np

try:
    import moderngl
except ImportError:  # pragma: no cover - exercised only in environments without moderngl
    moderngl = None

from ballast.render.camera import Camera
from ballast.render.meshes import Mesh

_VERTEX_SHADER = """
#version 330
uniform mat4 mvp;
uniform mat4 model;
in vec3 in_position;
in vec3 in_normal;
out vec3 v_normal;
out vec3 v_world_pos;
void main() {
    vec4 world_pos = model * vec4(in_position, 1.0);
    v_world_pos = world_pos.xyz;
    v_normal = mat3(model) * in_normal;
    gl_Position = mvp * vec4(in_position, 1.0);
}
"""

_FRAGMENT_SHADER = """
#version 330
uniform vec3 light_dir;
uniform vec3 base_color;
in vec3 v_normal;
in vec3 v_world_pos;
out vec4 f_color;
void main() {
    vec3 n = normalize(v_normal);
    float diffuse = max(dot(n, -light_dir), 0.0);
    float ambient = 0.25;
    vec3 specular_dir = normalize(reflect(light_dir, n));
    float spec = pow(max(dot(specular_dir, vec3(0.0, 0.0, 1.0)), 0.0), 16.0) * 0.2;
    vec3 color = base_color * (ambient + 0.75 * diffuse) + vec3(spec);
    f_color = vec4(color, 1.0);
}
"""


class GLRasterizer:
    """Standalone (headless) OpenGL rasterizer. Construct once, call
    `render` per frame; the context and compiled program are reused."""

    def __init__(self, width_px: int, height_px: int):
        if moderngl is None:
            raise RuntimeError("moderngl is not installed; use ballast.render.cpu_fallback instead")
        self.width_px = width_px
        self.height_px = height_px
        self.ctx = moderngl.create_context(standalone=True)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.program = self.ctx.program(vertex_shader=_VERTEX_SHADER, fragment_shader=_FRAGMENT_SHADER)
        self.color_texture = self.ctx.texture((width_px, height_px), 4, dtype="f1")
        self.depth_renderbuffer = self.ctx.depth_renderbuffer((width_px, height_px))
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.color_texture], depth_attachment=self.depth_renderbuffer
        )
        self._vao_cache: dict[int, tuple] = {}

    def _vao_for(self, mesh: Mesh):
        key = id(mesh)
        if key in self._vao_cache:
            return self._vao_cache[key]

        vertex_data = np.concatenate([mesh.vertices, mesh.normals], axis=1).astype("f4")
        vbo = self.ctx.buffer(vertex_data.tobytes())
        ibo = self.ctx.buffer(mesh.faces.astype("i4").tobytes())
        vao = self.ctx.vertex_array(self.program, [(vbo, "3f 3f", "in_position", "in_normal")], ibo)
        self._vao_cache[key] = (vao, vbo, ibo)
        return self._vao_cache[key]

    def render(
        self,
        mesh: Mesh,
        rotation: np.ndarray,
        translation: np.ndarray,
        camera: Camera,
        base_color: tuple[float, float, float] = (0.65, 0.62, 0.55),
        light_dir: tuple[float, float, float] = (0.4, 0.4, -0.8),
        background: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render one frame. Returns (rgb, mask): `rgb` is (H, W, 3) uint8,
        `mask` is (H, W) bool taken from the alpha channel (background alpha
        is 0, drawn geometry is opaque alpha 1)."""
        vao, _vbo, _ibo = self._vao_for(mesh)

        model = np.eye(4)
        model[:3, :3] = rotation
        model[:3, 3] = translation
        mvp = camera.projection_matrix() @ camera.view_matrix() @ model

        self.fbo.use()
        self.fbo.clear(*background)
        self.program["mvp"].write(mvp.T.astype("f4").tobytes())
        self.program["model"].write(model.T.astype("f4").tobytes())
        self.program["light_dir"].value = tuple(np.asarray(light_dir) / np.linalg.norm(light_dir))
        self.program["base_color"].value = base_color
        vao.render(moderngl.TRIANGLES)

        raw = np.frombuffer(self.color_texture.read(), dtype=np.uint8).reshape(
            self.height_px, self.width_px, 4
        )
        raw = np.flipud(raw)  # OpenGL reads bottom-up; flip to match image (row 0 = top) convention
        rgb = raw[:, :, :3]
        mask = raw[:, :, 3] > 0
        return rgb, mask

    def release(self) -> None:
        for vao, vbo, ibo in self._vao_cache.values():
            vao.release()
            vbo.release()
            ibo.release()
        self.fbo.release()
        self.color_texture.release()
        self.depth_renderbuffer.release()
        self.ctx.release()


def gl_available() -> bool:
    if moderngl is None:
        return False
    try:
        ctx = moderngl.create_context(standalone=True)
        ctx.release()
        return True
    except Exception:  # noqa: BLE001 - any GL init failure means "not available", not a bug to surface
        return False
