/**
 * Taken directly from  https://github.com/halide/Halide/blob/main/apps/blur/halide_blur_generator.cpp
 *  
 *  Minor changes:
 *  - Changed input type to float
 *  - Changed output type to float
 *  - Added printing to make sure we are hitting the GPU schedule.
 * 
 */

#include "Halide.h"

namespace {

enum class BlurGPUSchedule {
    Inline,          // Fully inlining schedule.
    Cache,           // Schedule caching intermedia result of blur_x.
    Slide,           // Schedule enabling sliding window opt within each
                     // work-item or cuda thread.
    SlideVectorize,  // The same as above plus vectorization per work-item.
    GernEquivalent,  // Equivalent schedule Gern. (I hope)
};

std::map<std::string, BlurGPUSchedule> blurGPUScheduleEnumMap() {
    return {
        {"inline", BlurGPUSchedule::Inline},
        {"cache", BlurGPUSchedule::Cache},
        {"slide", BlurGPUSchedule::Slide},
        {"slide_vector", BlurGPUSchedule::SlideVectorize},
        {"gern", BlurGPUSchedule::GernEquivalent},
    };
};

class HalideBlur : public Halide::Generator<HalideBlur> {
public:
    GeneratorParam<BlurGPUSchedule> schedule{
        "schedule",
        BlurGPUSchedule::Inline,
        blurGPUScheduleEnumMap()};
    GeneratorParam<int> tile_x{"tile_x", 32};  // X tile.
    GeneratorParam<int> tile_y{"tile_y", 8};   // Y tile.

    Input<Buffer<float, 2>> input{"input"};
    Output<Buffer<float, 2>> blur_y{"blur_y"};

    void generate() {
        Func blur_x("blur_x");
        Var x("x"), y("y"), xi("xi"), yi("yi"), xii("xii"), yii("yii");

        // The algorithm
        blur_x(x, y) = (input(x, y) + input(x + 1, y) + input(x + 2, y)) / 3;
        blur_y(x, y) = (blur_x(x, y) + blur_x(x, y + 1) + blur_x(x, y + 2)) / 3;

        // How to schedule it
        std::cout << get_target() << std::endl;
        if (get_target().has_gpu_feature()) {
            std::cout << "hit GPU schedule" << std::endl;
            // GPU schedule.
            switch (schedule) {
            case BlurGPUSchedule::Inline:
                std::cout << "BlurGPUSchedule::Inline" << std::endl;
                // - Fully inlining.
                blur_y.gpu_tile(x, y, xi, yi, tile_x, tile_y);
                break;
            case BlurGPUSchedule::Cache:
                std::cout << "BlurGPUSchedule::Cache" << std::endl;
                // - Cache blur_x calculation.
                blur_y.gpu_tile(x, y, xi, yi, tile_x, tile_y);
                blur_x.compute_at(blur_y, x).gpu_threads(x, y);
                break;
            case BlurGPUSchedule::Slide: {
                std::cout << "BlurGPUSchedule::Slide" << std::endl;
                // - Instead caching blur_x calculation explicitly, the
                //   alternative is to allow each work-item in OpenCL or thread
                //   in CUDA to calculate more rows of blur_y so that temporary
                //   blur_x calculation is re-used implicitly. This achieves
                //   the similar schedule of sliding window.
                Var y_inner("y_inner");
                blur_y
                    .split(y, y, y_inner, tile_y)
                    .reorder(y_inner, x)
                    .unroll(y_inner)
                    .gpu_tile(x, y, xi, yi, tile_x, 1);
                break;
            }
            case BlurGPUSchedule::SlideVectorize: {
                std::cout << "BlurGPUSchedule::SlideVectorize" << std::endl;
                // Vectorization factor.
                int factor = sizeof(int) / sizeof(short);
                Var y_inner("y_inner");
                blur_y.vectorize(x, factor)
                    .split(y, y, y_inner, tile_y)
                    .reorder(y_inner, x)
                    .unroll(y_inner)
                    .gpu_tile(x, y, xi, yi, tile_x, 1);
                break;
            }
            case BlurGPUSchedule::GernEquivalent: {
                std::cout << "BlurGPUSchedule::GernEquivalent" << std::endl;
                // Gern equivalent schedule.
                blur_y
                    .gpu_tile(x, y, xi, yi, 128, 128)
                    .tile(xi, yi, xii, yii, 8, 8, TailStrategy::RoundUp)
                    .vectorize(xii)
                    .unroll(yii);
                blur_y.dim(0).set_min(0).dim(1).set_stride((blur_y.width() / 8) * 8);
                break;
            }
            default:
                break;
            }
        } else if (get_target().has_feature(Target::HVX)) {
            std::cout << "hit HVX schedule" << std::endl;
            // Hexagon schedule.
            // TODO: Try using a schedule like the CPU one below.
            const int vector_size = 128;

            blur_y.compute_root()
                .hexagon()
                .prefetch(input, y, y, 2)
                .split(y, y, yi, 128)
                .parallel(y)
                .vectorize(x, vector_size * 2);
            blur_x
                .store_at(blur_y, y)
                .compute_at(blur_y, yi)
                .vectorize(x, vector_size);
        } else {
            std::cout << "hit CPU schedule" << std::endl;
            // CPU schedule.
            // Compute blur_x as needed at each vector of the output.
            // Halide will store blur_x in a circular buffer so its
            // results can be re-used.
            blur_y
                .split(y, y, yi, 32)
                .parallel(y)
                .vectorize(x, 16);
            blur_x
                .store_at(blur_y, y)
                .compute_at(blur_y, x)
                .vectorize(x, 16);
        }
    }
};

}  // namespace

HALIDE_REGISTER_GENERATOR(HalideBlur, halide_blur)