#ifndef GERN_UNCOPYABLE_H
#define GERN_UNCOPYABLE_H

namespace gern {
class Uncopyable {
protected:
  Uncopyable() = default;
  ~Uncopyable() = default;

private:
  Uncopyable(const Uncopyable &) = delete;
  Uncopyable &operator=(const Uncopyable &) = delete;
};
} // namespace gern

#endif
