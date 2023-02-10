#include "utils/graph/multidigraph.h"
#include "utils/hash-utils.h"

namespace FlexFlow {
namespace utils {

MultiDiEdge::MultiDiEdge(Node src, Node dst, size_t srcIdx, size_t dstIdx)
  : src(src), dst(dst), srcIdx(srcIdx), dstIdx(dstIdx)
{ }

typename MultiDiEdge::AsConstTuple MultiDiEdge::as_tuple() const {
  return {this->src, this->dst, this->srcIdx, this->dstIdx};
}

bool MultiDiEdge::operator==(MultiDiEdge const &other) const {
  return this->as_tuple() == other.as_tuple();
}

bool MultiDiEdge::operator<(MultiDiEdge const &other) const {
  return this->as_tuple() < other.as_tuple();
}

std::ostream &operator<<(std::ostream &s, MultiDiEdge const &e) {
  return (s << "MultiDiEdge<" << e.src.idx << ":" << e.srcIdx << " -> " << e.dst.idx << ":" << e.dstIdx << ">");
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_src_nodes(std::unordered_set<Node> const &nodes) const {
  MultiDiEdgeQuery e{*this};
  if (e.srcs != tl::nullopt) {
    throw std::runtime_error("expected srcs == tl::nullopt");
  }
  e.srcs = nodes;
  return e;
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_src_node(Node const &n) const {
  return this->with_src_nodes({n});
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_dst_nodes(std::unordered_set<Node> const &nodes) const {
  MultiDiEdgeQuery e{*this};
  if (e.dsts != tl::nullopt) {
    throw std::runtime_error("expected dsts == tl::nullopt");
  }
  e.dsts = nodes;
  return e;
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_dst_node(Node const &n) const {
  return this->with_dst_nodes({n});
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_src_idxs(std::unordered_set<std::size_t> const &idxs) const {
  MultiDiEdgeQuery e{*this};
  if (e.srcIdxs != tl::nullopt) {
    throw std::runtime_error("expected srcIdxs == tl::nullopt");
  }
  e.srcIdxs = idxs;
  return e;
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_src_idx(std::size_t idx) const {
  return this->with_src_idxs({idx});
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_dst_idxs(std::unordered_set<std::size_t> const &idxs) const {
  MultiDiEdgeQuery e{*this};
  if (e.dstIdxs != tl::nullopt) {
    throw std::runtime_error("expected dstIdxs == tl::nullopt");
  }
  e.dstIdxs = idxs;
  return e;
}

MultiDiEdgeQuery MultiDiEdgeQuery::with_dst_idx(std::size_t idx) const {
  return this->with_dst_idxs({idx});
}

MultiDiEdgeQuery MultiDiEdgeQuery::all() {
  return MultiDiEdgeQuery{};
}

}
}

namespace std {
using ::FlexFlow::utils::MultiDiEdge;

std::size_t hash<MultiDiEdge>::operator()(MultiDiEdge const &e) const {
  return get_std_hash(e.as_tuple());
}
}
