#ifndef GERN_RESOLVE_CONTRAINTS_H
#define GERN_RESOLVE_CONTRAINTS_H

#include <map>

#include "annotations/data_dependency_language.h"

namespace gern {
namespace resolve {

/**
 * @brief Given a system of equations, return the solution for
 *        each Variable.
 *
 * Currently, solve uses the GiNaC library. To plug-in a different
 * method for bounds inference, re-implement this function adhering to
 * same interface. There are no dependencies on GiNaC except in the
 * implementation of this function.
 *
 * @param system_of_equations The system of equations
 * @return std::map<Variable, Expr> Map from system of equations to
 *                                  solutions.
 */
std::map<Variable, Expr> solve(std::vector<Expr> system_of_equations);

} // namespace resolve
} // namespace gern

#endif