import argparse
import numpy as np
from pathlib import Path
import taichi as ti
from taichi.math import vec2, vec3, vec4, ivec2, ivec3, ivec4
import xarray as xr

from morton import decode_morton_3

TI_ARCH = ti.gpu
TI_FP = ti.f32

TRANSFER_CENTRE_VAL = 4
TRANSFER_CENTRE_WIDTH = 0.1
TRANSFER_BASE_COLOR = vec3(0.8, 0.0, 0.8) * 0.1
USE_HDDA = True
USE_MSAA = True

WINDOW_SIZE = (1920, 1080)
CAM_HEIGHT = 720
CAM_ASPECT = 16 / 9

BLUR_KERNEL_SIZE = 31


class OrthoCamera:
    def __init__(self, x_size=512, y_size=512, supersample=1):
        self.x_size = x_size
        self.y_size = y_size
        self.supersample = supersample
        self.centre = vec3(0.5 * x_size, 0.5 * y_size, 0.0)
        self.right_ray = vec3(1.0, 0.0, 0.0)
        self.up_ray = vec3(0.0, 0.0, 1.0)
        self.forward_ray = vec3(0.0, 1.0, 0.0)

    @property
    def corner(self):
        return self.centre - 0.5 * self.x_size * self.right_ray - 0.5 * self.y_size * self.up_ray

    def set_centre(self, centre: vec3):
        self.centre = centre
        return self

    def look_at(self, target):
        forward = target - self.centre
        forward /= forward.norm()
        default_up = vec3(0.0, 0.0, 1.0)
        if (forward - default_up).norm() < 1e-4:
            self.right_ray = vec3(0.0, -np.sign(forward[2]), 0.0)
        else:
            self.right_ray = default_up.cross(-forward)
            self.right_ray /= self.right_ray.norm()
        self.up_ray = (-forward).cross(self.right_ray)
        self.up_ray /= self.up_ray.norm()
        self.forward_ray = forward
        return self

def get_msaa_samples(sample_count: int):
    # https://vulkan.lunarg.com/doc/view/1.4.313.1/mac/antora/spec/latest/chapters/primsrast.html#primsrast-multisampling
    if sample_count not in [1, 2, 4, 8, 16]:
        raise ValueError("Unknown MSAA support requested")
    if sample_count == 1:
        samples = np.array([
            (0.5, 0.5)
        ])
    elif sample_count == 2:
        samples = np.array([
            (0.75, 0.75),
            (0.25, 0.25),
        ])
    elif sample_count == 4:
        samples = np.array([
            (0.375, 0.125),
            (0.875, 0.375),
            (0.125, 0.625),
            (0.625, 0.875),
        ])
    elif sample_count == 8:
        samples = np.array([
            (0.5625, 0.3125),
            (0.4375, 0.6875),
            (0.8125, 0.5625),
            (0.3125, 0.1875),
            (0.1875, 0.8125),
            (0.0625, 0.4375),
            (0.6875, 0.9375),
            (0.9375, 0.0625),
        ])
    elif sample_count == 16:
        samples = np.array([
            (0.5625, 0.5625),
            (0.4375, 0.3125),
            (0.3125, 0.625),
            (0.75, 0.4375),
            (0.1875, 0.375),
            (0.625, 0.8125),
            (0.8125, 0.6875),
            (0.6875, 0.1875),
            (0.375, 0.875),
            (0.5, 0.0625),
            (0.25, 0.125),
            (0.125, 0.75),
            (0.0, 0.5),
            (0.9375, 0.25),
            (0.875, 0.9375),
            (0.0625, 0.0),
        ])
    return samples.astype(np.float32)

RayHit = ti.types.struct(hit=ti.i32, t_range=vec2)
@ti.func
def intersect(bbox, origin, dir) -> RayHit:
    '''Check intersection with bounding box - returns Bool and contact distances
    Takes bounding box and direction from Slit class, origin of ray'''
    check = ti.i32(1)

    tmin = ti.f32(0.0)
    tmax = ti.f32(1.0e24)
    bbox_shrink = ti.f32(1e-4)
    ndim = 3

    for dim in range(ndim):
        if check == 0:
            continue
        inv_dir = 1.0 / dir[dim]
        #bbox assumued to be [0->x, 0->y, 0->z]
        #shrink bbox a little to avoid rays not being detected on the surface
        lower = 0.0 + bbox_shrink
        upper = bbox[dim] - bbox_shrink

        #entry and exit contact distances a and b
        a = (lower - origin[dim])*inv_dir
        b = (upper - origin[dim])*inv_dir

        #need to swap max and min incase of ray from other direction
        if a > b:
            a, b = b, a

        #closest entry and exit - comparing to previous dimension
        if a > tmin:
            tmin = a
        if b < tmax:
            tmax = b

        #total ray miss
        if tmin > tmax:
            check = ti.i32(0)

    return RayHit(hit=check, t_range=vec2(tmin,tmax))

@ti.dataclass
class HddaState:
    t: ti.f32
    dt: ti.f32
    step_size: ti.i32
    step_axis: ti.i32
    curr_coord: ivec3
    step: ivec3
    next_hit: vec3
    delta: vec3


    # NOTE(cmo): ti.template forces the s to be passed by reference rather than
    # needing to manage it with return values
    @ti.func
    def compute_axis_and_dt(self: ti.template(), ts):
        if (self.next_hit[0] <= self.next_hit[1] and self.next_hit[0] <= self.next_hit[2]):
            self.step_axis = 0
        elif (self.next_hit[1] <= self.next_hit[2]):
            self.step_axis = 1
        else:
            self.step_axis = 2

        next_t = self.next_hit[self.step_axis]
        if (next_t <= self.t):
            self.next_hit[self.step_axis] += self.t - 0.999999 * self.next_hit[self.step_axis] + 1e-6
            next_t = self.next_hit[self.step_axis]

        if next_t > ts[1]:
            self.dt = ts[1] - self.t
        else:
            self.dt = next_t - self.t
        self.dt = max(self.dt, 0.0)

    @ti.func
    def next_intersection(self: ti.template(), ts):
        self.t = self.next_hit[self.step_axis]
        self.next_hit[self.step_axis] += self.step_size * self.delta[self.step_axis]
        self.curr_coord[self.step_axis] += self.step_size * self.step[self.step_axis]
        self.compute_axis_and_dt(ts)
        return self.t < ts[1]

@ti.func
def HddaState_init():
    s = HddaState()
    s.t = 0.0
    s.dt = 0.0
    s.step_size = 0
    s.step_axis = 0
    s.curr_coord = ivec3(0)
    s.step = ivec3(0)
    s.next_hit = vec3(0.0)
    s.delta = vec3(0.0)
    return s


@ti.data_oriented
class TraceDex3d:
    def __init__(self, ds, param_name="temperature", transform_param=None):
        if ds.program != "dexrt (3d)":
            print("Program tag does not appear to be \"dexrt (3d)\", are you sure this is the right file?")
        if ds.output_format != "sparse":
            raise ValueError("Expected a sparse file")
        self.block_size = int(ds.block_size)
        self.block_stride = self.block_size**3
        self.log2_block_size = int(np.log2(self.block_size))
        self.num_x = int(ds.num_x)
        self.num_y = int(ds.num_y)
        self.num_z = int(ds.num_z)
        self.aabb = vec3(self.num_x, self.num_y, self.num_z)
        self.num_active_tiles = int(ds.num_active_tiles.shape[0])
        self.voxel_scale = float(ds.voxel_scale)
        self.offset_x = float(ds.offset_x)
        self.offset_y = float(ds.offset_y)
        self.offset_z = float(ds.offset_z)

        self.morton_tiles = ti.field(dtype=ti.u32, shape=(self.num_active_tiles,))
        self.param = ti.field(dtype=TI_FP, shape=(self.num_active_tiles * self.block_size**3,))
        self.morton_tiles.from_numpy(ds.morton_tiles.values)
        param_data = ds[param_name].values
        if transform_param is not None:
            param_data = transform_param(param_data)
        self.param.from_numpy(param_data)
        self.block_map = ti.field(dtype=ti.i32, shape=(self.num_z, self.num_y, self.num_x))
        self.setup_block_map()

    def setup_block_map(self):
        self.block_map.fill(-1)
        self.setup_block_map_kernel()

    @ti.kernel
    def setup_block_map_kernel(self):
        for tile_idx in self.morton_tiles:
            code = self.morton_tiles[tile_idx]
            block_coord = decode_morton_3(code)
            self.block_map[block_coord.z, block_coord.y, block_coord.x] = tile_idx

    @staticmethod
    @ti.func
    def transfer_function(param) -> vec3:
        return TRANSFER_BASE_COLOR * ti.exp(-(TRANSFER_CENTRE_VAL - param)**2 / TRANSFER_CENTRE_WIDTH)

    @ti.func
    def naive_sparse_trace(self, o: vec3, d: vec3, ts: vec2, step_size: ti.f32 = 0.2) -> vec3:
        t = ts[0]

        result = vec3(0.0)
        while t < ts[1]:
            sample_pos = o + t * d
            vox = ti.cast(sample_pos, ti.i32)
            block_coord = vox >> ivec3(self.log2_block_size)
            block_idx = self.block_map[block_coord.z, block_coord.y, block_coord.x]
            if block_idx != -1:
                inner_coord = vox - block_coord * self.block_size
                param = self.param[block_idx * self.block_stride + ((inner_coord.z * self.block_size + inner_coord.y) * self.block_size + inner_coord.x)]

                # Replace with RTE
                result += self.transfer_function(param) * self.voxel_scale * step_size

            t += step_size
        return result

    @ti.func
    def has_data(self, coord: ivec3) -> ti.i32:
        block_coord = coord >> ivec3(self.log2_block_size)
        block_idx = self.block_map[block_coord.z, block_coord.y, block_coord.x]
        return block_idx != -1

    @ti.func
    def step_through_grid(self, s: ti.template(), o, d, ts, inv_d):
        result = 0
        while not result and s.next_intersection(ts):
            has_data = self.has_data(s.curr_coord)

            if has_data and s.step_size == 1:
                result = 1
                break

            if not has_data and s.step_size != self.block_size:
                s.t += 0.01

            s.step_size = 1 if has_data else self.block_size
            curr_pos = o + s.t * d
            new_coord = ti.cast(curr_pos, ti.i32)
            s.curr_coord = new_coord & (~(s.step_size - 1))

            for ax in range(3):
                if s.step[ax] == 0:
                    continue

                s.next_hit[ax] = s.t + (s.curr_coord[ax] - curr_pos[ax]) * inv_d[ax]
                if s.step[ax] > 0:
                    s.next_hit[ax] += s.step_size * inv_d[ax]
            s.compute_axis_and_dt(ts)
            result = has_data

        return result

    @ti.func
    def two_level_hdda_trace(self, o: vec3, d: vec3, ts: vec2) -> vec3:
        s = HddaState_init()
        s.t = ts[0]

        start_pos = o + s.t * d
        s.curr_coord = ti.cast(start_pos, ti.i32)
        s.step_size = 1
        if not self.has_data(s.curr_coord):
            s.step_size = self.block_size
        s.curr_coord &= (~(s.step_size - 1))

        inv_d = 1.0 / d
        for ax in range(3):
            if d[ax] == 0.0:
                s.step[ax] = 0
                s.next_hit[ax] = 1e24
            elif (inv_d[ax] > 0.0):
                s.step[ax] = 1
                s.next_hit[ax] = s.t + (s.curr_coord[ax] + s.step_size - start_pos[ax]) * inv_d[ax]
                s.delta[ax] = inv_d[ax]
            else:
                s.step[ax] = -1
                s.next_hit[ax] = s.t + (s.curr_coord[ax] - start_pos[ax]) * inv_d[ax]
                s.delta[ax] = -inv_d[ax]
        s.compute_axis_and_dt(ts)

        result = vec3(0.0)
        while True:
            if self.has_data(s.curr_coord):
                block_coord = s.curr_coord >> ivec3(self.log2_block_size)
                block_idx = self.block_map[block_coord.z, block_coord.y, block_coord.x]
                if block_idx != -1:
                    inner_coord = s.curr_coord - block_coord * self.block_size
                    param = self.param[block_idx * self.block_stride + ((inner_coord.z * self.block_size + inner_coord.y) * self.block_size + inner_coord.x)]

                    # Replace with RTE
                    result += self.transfer_function(param) * self.voxel_scale * s.dt
            if not self.step_through_grid(s, o, d, ts, inv_d):
                break
        return result


    @ti.kernel
    def trace(self, supersample: int, corner: vec3, right: vec3, up: vec3, forward: vec3, step_size: ti.f32):
        num_aa_samples = cam.supersample if USE_MSAA else supersample**2
        for u, v in self.fb:
            radiance = vec3(0.0)
            ti.loop_config(serialize=True)
            for sub_ray in range(num_aa_samples):
                u_ray_offset, v_ray_offset = 0.5, 0.5
                if USE_MSAA:
                    u_ray_offset = msaa_samples[sub_ray][0]
                    v_ray_offset = msaa_samples[sub_ray][1]
                else:
                    u_ray = sub_ray // supersample
                    v_ray = sub_ray - u_ray * supersample
                    u_ray_offset = (u_ray + 0.5) / ti.f32(supersample)
                    v_ray_offset = (v_ray + 0.5) / ti.f32(supersample)
                ray_start_uvw = corner + right * (u + u_ray_offset) + up * (v + v_ray_offset)

                ray_hit = intersect(self.aabb, origin=ray_start_uvw, dir=forward)
                sample_color = vec3(0.0)

                if ray_hit.hit:
                    if USE_HDDA:
                        sample_color = self.two_level_hdda_trace(
                            o=ray_start_uvw,
                            d=forward,
                            ts=ray_hit.t_range,
                        )
                    else:
                        sample_color = self.naive_sparse_trace(
                            o=ray_start_uvw,
                            d=forward,
                            ts=ray_hit.t_range,
                            step_size=step_size,
                        )

                radiance += sample_color
            self.fb[u, v] = radiance / num_aa_samples


    def trace_for_camera(self, cam: OrthoCamera, step_size=0.2):
        if not hasattr(self, 'fb') or self.fb.shape != (cam.x_size, cam.y_size):
            self.fb = ti.field(vec3, shape=(cam.x_size, cam.y_size))
        self.trace(
            supersample=cam.supersample,
            corner=cam.corner,
            right=cam.right_ray,
            up=cam.up_ray,
            forward=cam.forward_ray,
            step_size=step_size,
        )

@ti.func
def tonemap(x: vec3) -> vec3:
    """Uncharted2 tonemap"""
    A = 0.15
    B = 0.5
    C = 0.1
    D = 0.2
    E = 0.02
    F = 0.3

    return ((x * (A*x + C*B) + D*E) / (x * (A*x + B) + D*F)) - E/F


def tonemap_and_blit(trace_buffer, bias, white_point, blur):
    if blur > 1:
        compute_blur_kernel(blur)
        blur_pass_horizontal(trace_buffer, blur_buffer)
        blur_pass_vertical(blur_buffer, trace_buffer)
    tonemap_buffer(trace_buffer, bias, white_point)
    if gui_buffer.shape == trace_buffer.shape:
        blit(trace_buffer)
    else:
        bilinear_copy(trace_buffer)

@ti.kernel
def tonemap_buffer(trace_buffer: ti.template(), bias: ti.f32, white_scale: ti.f32):
    for i, j in trace_buffer:
        sample = (tonemap(bias * trace_buffer[i, j]) * white_scale)**(1.0 / 2.2)
        trace_buffer[i, j] = sample

@ti.kernel
def blit(trace_buffer: ti.template()):
    for u, v in trace_buffer:
        gui_buffer[u, v] = trace_buffer[u, v]

@ti.kernel
def bilinear_copy(trace_buffer: ti.template()):
    # NOTE(cmo): Only uses height ratio
    ratio = trace_buffer.shape[1] / gui_buffer.shape[1]
    for u, v in gui_buffer:
        uprime = u * ratio
        vprime = v * ratio
        if uprime >= trace_buffer.shape[0] or vprime >= trace_buffer.shape[1]:
            gui_buffer[u, v] = vec3(1.0, 0.0, 1.0)
            continue
        uv = vec2(uprime, vprime)
        corner = ti.cast(ti.math.floor(uv), ti.i32)
        bilin_ratio = ti.math.fract(uv)
        weights = vec4(
            (1.0 - bilin_ratio.x) * (1.0 - bilin_ratio.y),
            (1.0 - bilin_ratio.x) * bilin_ratio.y,
            bilin_ratio.x * (1.0 - bilin_ratio.y),
            bilin_ratio.x * bilin_ratio.y,
        )
        result = vec3(0.0)
        for p in range(4):
            py = p // 2
            px = p - py * 2
            result += weights[p] * trace_buffer[corner[0] + px, corner[1] + py]
        gui_buffer[u, v] = result

@ti.kernel
def compute_blur_kernel(fwhm: ti.f32):
    sigma = fwhm / 2.35482
    mu = blur_kernel.shape[0] // 2
    for i in blur_kernel:
        blur_kernel[i] = 1.0 / (sigma * ti.sqrt(2 * ti.math.pi)) * ti.exp(-0.5 * (i - mu)**2 / sigma**2)

@ti.kernel
def blur_pass_horizontal(buffer: ti.template(), out_buffer: ti.template()):
    offset = blur_kernel.shape[0] // 2
    for u, v in buffer:
        out_buffer[u, v] = 0.0
        for uu in range(blur_kernel.shape[0]):
            u_idx = u - offset + uu
            if u_idx < 0 or u_idx >= buffer.shape[0]:
                continue
            out_buffer[u, v] += blur_kernel[uu] * buffer[u_idx, v]

@ti.kernel
def blur_pass_vertical(buffer: ti.template(), out_buffer: ti.template()):
    offset = blur_kernel.shape[0] // 2
    for u, v in buffer:
        out_buffer[u, v] = 0.0
        for vv in range(blur_kernel.shape[0]):
            v_idx = v - offset + vv
            if v_idx < 0 or v_idx >= buffer.shape[1]:
                continue
            out_buffer[u, v] += blur_kernel[vv] * buffer[u, v_idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trace 3D volume from sparse Dex atmosphere"
        )
    parser.add_argument(
        '--path',
        metavar='PATH',
        type=Path,
        help="The path to the netcdf dex atmosphere",
    )
    args = parser.parse_args()
    ds = xr.open_dataset(args.path)

    ti.init(arch=TI_ARCH, default_fp=TI_FP)

    @ti.data_oriented
    class TraceDex3dDoubleTransfer(TraceDex3d):

        @staticmethod
        @ti.func
        def transfer_function(param):
            scale = 0.005
            return (
                scale * vec3(0.0, 0.0, 1.0) * ti.exp(-(param - 3.8)**2 / 0.1**2)
                + scale * vec3(0.0, 1.0, 0.0) * ti.exp(-(param - 4.5)**2 / 0.1**2)
                + scale * vec3(1.0, 0.0, 0.0) * ti.exp(-(param - 5)**2 / 0.1**2)
            )

    tracer = TraceDex3dDoubleTransfer(
        ds=ds,
        param_name="temperature",
        transform_param=lambda x: np.log10(x),
    )
    cam = OrthoCamera(supersample=8, x_size=int(CAM_ASPECT * CAM_HEIGHT), y_size=CAM_HEIGHT)
    if USE_MSAA:
        msaa_samples = ti.field(vec2, shape=(cam.supersample,))
        samples = get_msaa_samples(cam.supersample)
        msaa_samples.from_numpy(samples)
    else:
        msaa_samples = ti.field(vec2, shape=(1,))


    gui = ti.GUI("TraceTime", WINDOW_SIZE)
    gui_buffer = ti.field(vec3, shape=WINDOW_SIZE)

    theta_slider = gui.slider("Theta", 0.0, 360.0, step=1)
    phi_slider = gui.slider("Phi", 0.0, 180.0, step=1)
    phi_slider.value = 90.0
    bias = gui.slider("inv bias", 1.0, 200.0)
    white_point = gui.slider("white point", 0.5, 20.0)
    white_point.value = 1.0
    blur_width = gui.slider("PSF FWHM", 0.0, BLUR_KERNEL_SIZE // 2)
    cam_dist = 1e3
    blur_buffer = ti.field(vec3, shape=(cam.x_size, cam.y_size))
    blur_kernel = ti.field(ti.f32, shape=(BLUR_KERNEL_SIZE))
    while gui.running:
        theta = np.deg2rad(theta_slider.value)
        phi = np.deg2rad(phi_slider.value)
        r = vec3(np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi))
        cam.set_centre(tracer.aabb // 2 + r * cam_dist).look_at(tracer.aabb // 2)

        tracer.trace_for_camera(
            cam=cam,
            step_size=1,
        )

        tonemap_and_blit(tracer.fb, 1.0 / bias.value, white_point.value, blur_width.value)
        gui.set_image(gui_buffer)
        gui.show()