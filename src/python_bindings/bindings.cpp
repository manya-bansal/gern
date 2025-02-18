#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "compose/composable.h"
#include "library/matrix/annot/cpu-matrix.h"
#include "library/matrix/impl/cpu-matrix.h"
#include "test-utils.h"

namespace py = pybind11;
using namespace gern;

class MyInt {
public:
	MyInt(int64_t v) {
		val = new int64_t(v);
	}
	int64_t* val;
};

PYBIND11_MODULE(gern_py, m) {
	py::class_<Composable>(m, "Composable")
		.def(py::init<std::vector<Composable>>());
	
	py::class_<TileDummy>(m, "TileDummy")
		.def("__call__", [](TileDummy t, py::args args){
			std::vector<Composable> to_compose;
			for (auto arg : args) {
				to_compose.push_back(arg.cast<Composable>());
			}
			return t.operator()(Composable(new const Computation(to_compose)));
		});

	py::class_<ADTMember>(m, "ADTMember");

	py::class_<Datatype>(m, "DatatypeClass")
		.def(py::init<Datatype::Kind>());
	py::enum_<Datatype::Kind>(m, "Datatype")
		.value("Int64", Datatype::Int64)
		.value("Float32", Datatype::Float32)
		.export_values();
	py::class_<Variable>(m, "Variable")
		.def_static("init", [](const std::string &name,
             Datatype type = Datatype::Int64,
             bool const_expr = false){
				return new Variable(name, type, const_expr);
			 },
			 py::return_value_policy::reference,
			py::arg("name") = "a",
			py::arg("type") = Datatype(Datatype::Int64),
			py::arg("const_expr") = false)
		.def("getName", &Variable::getName)
		.def("bind", &Variable::bind);

	py::class_<AbstractDataType>(m, "AbstractDataType");

	py::class_<annot::MatrixCPU, AbstractDataType>(m, "AnnotMatrixCPU")
		.def("init", [](std::string &name){
			return new const annot::MatrixCPU(name);
		}, py::return_value_policy::reference);
	py::class_<AbstractDataTypePtr>(m, "AbstractDataTypePtr")
		.def(py::init<const AbstractDataType *>())
		.def("__getitem__", [](const AbstractDataTypePtr &self, std::string member) {
			return self[member];
		}, py::is_operator())
		.def("getName", &AbstractDataTypePtr::getName);
		

	m.def("Tile", &Tile);

	py::class_<Runner>(m, "Runner")
		.def(py::init<Composable>())
		.def("compile", &Runner::compile)
		.def("evaluate", [](Runner &self, std::map<std::string, void *> args){
			self.evaluate(args);
			// for (auto [key, val] : args) {
			// 	std::cout << key << " " << val << std::endl;
			// }
			// self.evaluate(args);
		});
	
	py::class_<Runner::Options>(m, "RunnerOptions");

	m.def("cpuRunner", py::overload_cast<const std::vector<std::string> &>(&test::cpuRunner));

	// py::class_<AbstractFunction>(m, "AbstractFunction")
	// 	.def("__call__", [](py::args args){
	// 		// rewrite AbstractFunction::operator() to work with py::args
	// 		std::vector<Argument> arguments;
	// 		for (auto arg : args) {
	// 			arguments.push_back(arg.cast<Argument>());
	// 		}
	// 	});
	// py::class_<annot::MatrixAddCPU, AbstractFunction>(m, "MatrixAddCPU")
	// 	.def(py::init<>());


	m.def("MatrixAddCPU", [](AbstractDataTypePtr in, AbstractDataTypePtr out, const std::map<std::string, Variable> &replacements = {}){
		annot::MatrixAddCPU add;
		if (!replacements.empty()) {
			add[replacements];
		}
		return add(in, out);
	}, py::arg(), py::arg(), py::arg("replacements") = std::map<std::string, Variable>{});

	m.def("MatrixSoftmax", [](AbstractDataTypePtr in, AbstractDataTypePtr out, const std::map<std::string, Variable> &replacements = {}){
		annot::MatrixSoftmax softmax;
		if (!replacements.empty()) {
			softmax[replacements];
		}
		return softmax(in, out);
	}, py::arg(), py::arg(), py::arg("replacements") = std::map<std::string, Variable>{});

	py::class_<impl::MatrixCPU>(m, "MatrixCPU")
		.def("init", [](uintptr_t ptr, int64_t row, int64_t col, int64_t lda){
			float* data = reinterpret_cast<float*>(ptr);
			return new impl::MatrixCPU(data, row, col, lda);
		}, py::return_value_policy::reference)
		.def("init", [](int64_t row, int64_t col, int64_t lda){
			auto mat = new impl::MatrixCPU(row, col, lda);
			return mat;
		}, py::return_value_policy::reference)
		.def("vvals", &impl::MatrixCPU::vvals)
		.def("random_fill", &impl::MatrixCPU::random_fill, py::arg("min") = 0.0f, py::arg("max") = 1.0f)
		.def("ascending", &impl::MatrixCPU::ascending)
		.def("__repr__", [](const impl::MatrixCPU &obj) {
            std::ostringstream oss;
            oss << obj;
            return oss.str();
        });
	
	m.def("getAddress", [](impl::MatrixCPU* mat){
		return static_cast<void*>(mat);
	});

	py::class_<MyInt>(m, "Int")
		.def("init", [](int64_t val){ return new MyInt(val); }, py::return_value_policy::reference);

	m.def("getAddress", [](MyInt* num){
		return static_cast<void*>(num->val);
	});

	m.def("getAddress", [](int64_t* num){
		return static_cast<void*>(num);
	});
	
}