#ifndef _FLEXFLOW_DEDUPLICATED_PRIORITY_QUEUE_H
#define _FLEXFLOW_DEDUPLICATED_PRIORITY_QUEUE_H

namespace FlexFlow {
    
template <typename Elem,
          typename Container = std::vector<Elem>,
          typename Compare = std::less<typename Container::value_type>,
          typename Hash = std::hash<Elem>>
class DeduplicatedPriorityQueue {
public:
  Elem const &top() const {
    return impl.top();
  }

  bool empty() const {
    return impl.empty();
  }

  size_t size() const {
    return impl.size();
  }

  void push(Elem const &e) {
    size_t hash = Hash{}(e);
    if (!contains(hashmap, e)) {
      impl.push(e);
      hashmap.insert(hash);
    }
  }

  void pop() {
    hashmap.erase(Hash{}(impl.top()));
    impl.pop();
  }

private:
  std::priority_queue<Elem, Container, Compare> impl;
  std::unordered_set<size_t> hashmap;
};

}


#endif /* _FLEXFLOW_DEDUPLICATED_PRIORITY_QUEUE_H */
