#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "compose/composable.h"
#include "library/matrix/annot/cpu-matrix.h"
#include "library/matrix/annot/cpu-matrix-template.h"
#include "library/matrix/impl/cpu-matrix.h"
#include "library/matrix/impl/cpu-matrix-template.h"
#include "test-utils.h"
#include "annotations/argument.h"

namespace py = pybind11;
using namespace gern;

class MyInt {
public:
	MyInt(int64_t v) {
		val = new int64_t(v);
	}
	int64_t* val;
};

class MyFloat {
public:
	MyFloat(float v) {
		val = new float(v);
	}
	float* val;
};

PYBIND11_MODULE(gern_py, m) {
	py::class_<Composable>(m, "Composable")
		.def(py::init<std::vector<Composable>>())
		.def("getAnnotation", &Composable::getAnnotation);
	
	py::class_<TileDummy>(m, "TileDummy")
		.def("__call__", [](TileDummy t, py::args args){
			std::vector<Composable> to_compose;
			for (auto arg : args) {
				to_compose.push_back(arg.cast<Composable>());
			}
			if (to_compose.size() == 1) {
				return t.operator()(to_compose[0]);
			}
			return t.operator()(Composable(new const Computation(to_compose)));
		});

	py::class_<Expr>(m, "Expr");
	py::class_<ADTMember, Expr>(m, "ADTMember")
		.def("getMember", &ADTMember::getMember);
	py::class_<Annotation>(m, "Annotation")
		.def("getPattern", &Annotation::getPattern);
	py::class_<gern::Pattern>(m, "Pattern")
		.def("getTileableFields", &Annotation::getTileableFields);

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
		.def("bind", &Variable::bind)
		.def("__repr__", [](const Variable &self){
			return self.getName();
		});

	py::class_<AbstractDataType>(m, "AbstractDataType");

	py::class_<annot::MatrixCPU, AbstractDataType>(m, "AnnotMatrixCPU")
		.def("init", [](std::string &name){
			return new const annot::MatrixCPU(name);
		}, py::return_value_policy::reference);
	
	py::class_<annot::MatrixCPU4Dim, AbstractDataType>(m, "AnnotMatrixCPU4Dim")
		.def("init", [](std::string &name){
			return new const annot::MatrixCPU4Dim(name);
		}, py::return_value_policy::reference);
	
	py::class_<annot::MatrixCPUTemplate, AbstractDataType>(m, "AnnotMatrixCPUTemplate")
		.def("init", [](std::string &name){
			return new const annot::MatrixCPUTemplate(name);
		}, py::return_value_policy::reference);

	py::class_<AbstractDataTypePtr>(m, "AbstractDataTypePtr")
		.def(py::init<const AbstractDataType *>())
		.def("__getitem__", [](const AbstractDataTypePtr &self, std::string member) {
			return self[member];
		}, py::is_operator())
		.def("getName", &AbstractDataTypePtr::getName)
		.def("__repr__", [](const AbstractDataTypePtr &self) {
			return self.str();
		});
		

	m.def("Tile", &Tile);

	py::class_<Runner>(m, "Runner")
		.def(py::init<Composable>())
		.def("compile", &Runner::compile)
		.def("evaluate", &Runner::evaluate);
	
	py::class_<Runner::Options>(m, "RunnerOptions");

	m.def("cpuRunner", py::overload_cast<const std::vector<std::string> &>(&test::cpuRunner));

	/// ANNOTATIONS
	py::class_<AbstractFunction>(m, "AbstractFunction")
		.def("__call__", [](AbstractFunction &self, py::args args) {
			std::vector<Argument> arguments;
			for (auto arg : args) {
				if (py::isinstance<AbstractDataTypePtr>(arg)) {
					self.addArg<AbstractDataTypePtr>(arguments, arg.cast<AbstractDataTypePtr>());
				} else if (py::isinstance<Variable>(arg)) {
					self.addArg<Variable>(arguments, arg.cast<Variable>());
				}
			}
			return self.constructComposable(arguments);
		})
		.def("setBindings", &AbstractFunction::operator[]);
	
	py::class_<annot::MatrixAddStaticStore, AbstractFunction>(m, "MatrixAddStaticStore")
		.def(py::init<>());

	py::class_<annot::MatrixAddCPU, AbstractFunction>(m, "MatrixAddCPU")
		.def(py::init<>());

	py::class_<annot::MatrixSoftmax, AbstractFunction>(m, "MatrixSoftmax")
		.def(py::init<>());

	py::class_<annot::MatrixTranspose, AbstractFunction>(m, "MatrixTranspose")
		.def(py::init<>());
	
	py::class_<annot::MatrixMultiply, AbstractFunction>(m, "MatrixMultiply")
		.def(py::init<>());
	
	py::class_<annot::MatrixDivn, AbstractFunction>(m, "MatrixDivn")
		.def(py::init<>());
	
	py::class_<annot::MatrixAttention, AbstractFunction>(m, "MatrixAttention")
		.def(py::init<>());
	
	py::class_<annot::MatrixAttention4D, AbstractFunction>(m, "MatrixAttention4D")
		.def(py::init<>());

	// m.def("MatrixAddCPU", [](AbstractDataTypePtr in, AbstractDataTypePtr out, const std::map<std::string, Variable> &replacements = {}){
	// 	annot::MatrixAddCPU add;
	// 	if (!replacements.empty()) {
	// 		add[replacements];
	// 	}
	// 	return add(in, out);
	// }, py::arg(), py::arg(), py::arg("replacements") = std::map<std::string, Variable>{});

	// m.def("MatrixSoftmax", [](AbstractDataTypePtr in, AbstractDataTypePtr out, const std::map<std::string, Variable> &replacements = {}){
	// 	annot::MatrixSoftmax softmax;
	// 	if (!replacements.empty()) {
	// 		softmax[replacements];
	// 	}
	// 	return softmax(in, out);
	// }, py::arg(), py::arg(), py::arg("replacements") = std::map<std::string, Variable>{});

	// m.def("MatrixTranspose", [](AbstractDataTypePtr in, AbstractDataTypePtr out, const std::map<std::string, Variable> &replacements = {}){
	// 	annot::MatrixTranspose transpose;
	// 	if (!replacements.empty()) {
	// 		transpose[replacements];
	// 	}
	// 	return transpose(in, out);
	// }, py::arg(), py::arg(), py::arg("replacements") = std::map<std::string, Variable>{});

	// m.def("MatrixMultiply", [](AbstractDataTypePtr a, AbstractDataTypePtr b, AbstractDataTypePtr out, const std::map<std::string, Variable> &replacements = {}){
	// 	annot::MatrixMultiply matmul;
	// 	if (!replacements.empty()) {
	// 		matmul[replacements];
	// 	}
	// 	return matmul(a, b, out);
	// }, py::arg(), py::arg(), py::arg(), py::arg("replacements") = std::map<std::string, Variable>{});

	// m.def("MatrixDivn", [](AbstractDataTypePtr in, Variable n, AbstractDataTypePtr out, const std::map<std::string, Variable> &replacements = {}){
	// 	annot::MatrixDivn divn;
	// 	if (!replacements.empty()) {
	// 		divn[replacements];
	// 	}
	// 	return divn(in, n, out);
	// }, py::arg(), py::arg(), py::arg(), py::arg("replacements") = std::map<std::string, Variable>{});

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

	py::class_<impl::MatrixCPU4Dim>(m, "MatrixCPU4Dim")
		.def("init", [](uintptr_t ptr, int64_t i_dim, int64_t j_dim, int64_t k_dim, int64_t l_dim, int64_t i_incr, int64_t j_incr, int64_t k_incr){
			float* data = reinterpret_cast<float*>(ptr);
			return new impl::MatrixCPU4Dim(data, i_dim, j_dim, k_dim, l_dim, i_incr, j_incr, k_incr);
		}, py::return_value_policy::reference)
		.def("init", [](int64_t i_dim, int64_t j_dim, int64_t k_dim, int64_t l_dim, int64_t i_incr, int64_t j_incr, int64_t k_incr){
			auto mat = new impl::MatrixCPU4Dim(i_dim, j_dim, k_dim, l_dim, i_incr, j_incr, k_incr);
			return mat;
		}, py::return_value_policy::reference)
		.def("vvals", &impl::MatrixCPU4Dim::vvals)
		.def("random_fill", &impl::MatrixCPU4Dim::random_fill, py::arg("min") = 0.0f, py::arg("max") = 1.0f)
		.def("ascending", &impl::MatrixCPU4Dim::ascending);

	// helper classes to create variable values in c++
	// and give c++ ownership (to get pointers later)
	py::class_<MyInt>(m, "Int")
		.def("init", [](int64_t val){ return new MyInt(val); }, py::return_value_policy::reference)
		.def("__repr__", [](const MyInt &obj) {
			std::ostringstream oss;
			oss << *obj.val;
			return oss.str();
		});

	py::class_<MyFloat>(m, "Float")
		.def("init", [](float val){ return new MyFloat(val); }, py::return_value_policy::reference)
		.def("__repr__", [](const MyFloat &obj) {
			std::ostringstream oss;
			oss << *obj.val;
			return oss.str();
		});
	
	m.def("getAddress", [](impl::MatrixCPU4Dim* mat){
		return static_cast<void*>(mat);
	});
	
	m.def("getAddress", [](impl::MatrixCPU* mat){
		return static_cast<void*>(mat);
	});

	m.def("getAddress", [](MyInt* num){
		return static_cast<void*>(num->val);
	});

	m.def("getAddress", [](MyFloat* num){
		return static_cast<void*>(num->val);
	});
}