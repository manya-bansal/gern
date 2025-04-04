#pragma once

#include <map>

#include "annotations/data_dependency_language.h"

namespace gern {
namespace resolve {

/**
 * @brief Given a system of equations, return the solution for
 *        each Variable.
 *
 * Currently, solve uses the GiNaC library. To plug-in a different
 * method for bounds inference, re-implement this FunctionSignature adhering to
 * same interface. There are no dependencies on GiNaC except in the
 * implementation of this function.
 *
 * @param system_of_equations The system of equations
 * @return std::map<Variable, Expr> Map from system of equations to
 *                                  solutions.
 */
Expr solve(Eq eq, Variable v);

}  // namespace resolve
}  // namespace gern