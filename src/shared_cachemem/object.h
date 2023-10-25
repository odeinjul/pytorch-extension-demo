#ifndef OBJECT_H_
#define OBJECT_H_

/*
  from: dgl/object.h, dmlc/logging.h
*/

#include <iostream>
#include <sstream>
#include <string>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <sys/mman.h>

#define CHECK_BINARY_OP(name, op, x, y)                  \
  if (auto __log__err = LogCheck##name(x, y))  \
      std::cout << "[ERROR] Check failed: " << #x " " #op " " #y << *__log__err << ": "


#define CHECK_NOTNULL(x) \
  ((x) == NULL ? std::cout << "[ERROR] Check notnull: " <<  #x << ' ', (x) : (x)) // NOLINT(*)
  //((x) == NULL ? dmlc::LogMessageFatal(__FILE__, __LINE__).stream() << "Check  notnull: "  #x << ' ', (x) : (x)) // NOLINT(*)

#define CHECK(x)         \
  if (!(x)) std::cout << "[ERROR] Check failed: " << #x << ": "

#define CHECK_NE(x, y) CHECK_BINARY_OP(_NE, !=, x, y)

template <typename X, typename Y>
std::unique_ptr<std::string> LogCheckFormat(const X& x, const Y& y) {
  std::ostringstream os;
  os << " (" << x << " vs. " << y << ") "; /* CHECK_XX(x, y) requires x and y can be serialized to string. Use CHECK(x OP y) otherwise. NOLINT(*) */
  // no std::make_unique until c++14
  return std::unique_ptr<std::string>(new std::string(os.str()));
}

template <typename X, typename Y>                                                        \
inline std::unique_ptr<std::string> LogCheck_NE(const X& x, const Y& y) { \
  if (x != y) return nullptr;                                                            \
  return LogCheckFormat(x, y);                                                           \
}                                                                                        \
inline std::unique_ptr<std::string> LogCheck_NE(int x, int y) {           \
  return LogCheck_NE<int, int>(x, y);                                                 \
}

class Object;
class ObjectRef;
//class NDArray;

// single manager of operator information.
struct TypeManager {
  // mutex to avoid registration from multiple threads.
  // recursive is needed for trigger(which calls UpdateAttrMap)
  std::mutex mutex;
  std::atomic<uint32_t> type_counter{0};
  std::unordered_map<std::string, uint32_t> key2index;
  std::vector<std::string> index2key;
  // get singleton of the
  static TypeManager* Global() {
    static TypeManager inst;
    return &inst;
  }
};


/*!
 * \brief Visitor class to each object attribute.
 *  The content is going to be called for each field.
 */
/*
class AttrVisitor {
 public:
//! \cond Doxygen_Suppress
  virtual void Visit(const char* key, double* value) = 0;
  virtual void Visit(const char* key, int64_t* value) = 0;
  virtual void Visit(const char* key, uint64_t* value) = 0;
  virtual void Visit(const char* key, int* value) = 0;
  virtual void Visit(const char* key, bool* value) = 0;
  virtual void Visit(const char* key, std::string* value) = 0;
  virtual void Visit(const char* key, ObjectRef* value) = 0;
  virtual void Visit(const char* key, NDArray* value) = 0;
  template<typename ENum,
           typename = typename std::enable_if<std::is_enum<ENum>::value>::type>
  void Visit(const char* key, ENum* ptr) {
    static_assert(std::is_same<int, typename std::underlying_type<ENum>::type>::value,
                  "declare enum to be enum int to use visitor");
    this->Visit(key, reinterpret_cast<int*>(ptr));
  }
//! \endcond
};
*/

/*!
 * \brief base class of object container.
 *  All object's internal is stored as std::shared_ptr<Object>
 */
class Object {
 public:
  /*! \brief virtual destructor */
  virtual ~Object() {}
  /*! \return The unique type key of the object */
  virtual const char* type_key() const = 0;
  /*!
   * \brief Apply visitor to each field of the Object
   *  Visitor could mutate the content of the object.
   *  override if Object contains attribute fields.
   * \param visitor The visitor
   */

  //TODO
  //virtual void VisitAttrs(AttrVisitor* visitor) {}
  /*! \return the type index of the object */

  virtual uint32_t type_index() const = 0;
  /*!
   * \brief Whether this object derives from object with type_index=tid.
   *  Implemented by DGL_DECLARE_OBJECT_TYPE_INFO
   *
   * \param tid The type index.
   * \return the check result.
   */
  virtual bool _DerivedFrom(uint32_t tid) const;
  /*!
   * \brief get a runtime unique type index given a type key
   * \param type_key Type key of a type.
   * \return the corresponding type index.
   */
  static uint32_t TypeKey2Index(const char* type_key);
  /*!
   * \brief get type key from type index.
   * \param index The type index
   * \return the corresponding type key.
   */
  static const char* TypeIndex2Key(uint32_t index);
  /*!
   * \return whether the type is derived from
   */
  template<typename T>
  inline bool derived_from() const;
  /*!
   * \return whether the object is of type T
   * \tparam The type to be checked.
   */
  template<typename T>
  inline bool is_type() const;
  // object ref can see this
  friend class ObjectRef;
  static constexpr const char* _type_key = "Object";
};


/*! \brief base class of all reference object */
class ObjectRef {
 public:
  /*! \brief type indicate the container type */
  using ContainerType = Object;
  /*!
   * \brief Comparator
   *
   * Compare with the two are referencing to the same object (compare by address).
   *
   * \param other Another object ref.
   * \return the compare result.
   * \sa same_as
   */
  inline bool operator==(const ObjectRef& other) const;
  /*!
   * \brief Comparator
   *
   * Compare with the two are referencing to the same object (compare by address).
   *
   * \param other Another object ref.
   * \return the compare result.
   */
  inline bool same_as(const ObjectRef& other) const;
  /*!
   * \brief Comparator
   *
   * The operator overload allows ObjectRef be used in std::map.
   *
   * \param other Another object ref.
   * \return the compare result.
   */
  inline bool operator<(const ObjectRef& other) const;
  /*!
   * \brief Comparator
   * \param other Another object ref.
   * \return the compare result.
   * \sa same_as
   */
  inline bool operator!=(const ObjectRef& other) const;
  /*! \return the hash function for ObjectRef */
  inline size_t hash() const;
  /*! \return whether the expression is null */
  inline bool defined() const;
  /*! \return the internal type index of Object */
  inline uint32_t type_index() const;
  /*! \return the internal object pointer */
  inline const Object* get() const;
  /*! \return the internal object pointer */
  inline const Object* operator->() const;
  /*!
   * \brief Downcast this object to its actual type.
   * This returns nullptr if the object is not of the requested type.
   * Example usage:
   *
   * if (const Banana *banana = obj->as<Banana>()) {
   *   // This is a Banana!
   * }
   * \tparam T the target type, must be subtype of Object
   */
  template<typename T>
  inline const T *as() const;

  /*! \brief default constructor */
  ObjectRef() = default;
  explicit ObjectRef(std::shared_ptr<Object> obj) : obj_(obj) {}

  /*! \brief the internal object, do not touch */
  std::shared_ptr<Object> obj_;
};


/*
 * \brief This class owns shared memory.
 *
 * When the object is gone, the shared memory will also be destroyed.
 * When the shared memory is destroyed, the file corresponding to
 * the shared memory is removed.
 */
class SharedMemory {
  /*
   * \brief whether the shared memory is owned by the object.
   *
   * If shared memory is created in the object, it'll be owned by the object
   * and will be responsible for deleting it when the object is destroyed.
   */
  bool own;

  /* \brief the file descripter of the shared memory. */
  int fd;
  /* \brief the address of the shared memory. */
  void *ptr;
  /* \brief the size of the shared memory. */
  size_t size;

  /*
   * \brief the name of the object.
   *
   * In Unix, shared memory is identified by a file. Thus, `name` is actually
   * the file name that identifies the shared memory.
   */
  std::string name;

 public:
  /* \brief Get the filename of shared memory file
   */
  std::string GetName() const { return name; }

  /*
   * \brief constructor of the shared memory.
   * \param name The file corresponding to the shared memory.
   */
  explicit SharedMemory(const std::string &name);
  /*
   * \brief destructor of the shared memory.
   * It deallocates the shared memory and removes the corresponding file.
   */
  ~SharedMemory();
  /*
   * \brief create shared memory.
   * It creates the file and shared memory.
   * \param size the size of the shared memory.
   * \return the address of the shared memory
   */
  void *CreateNew(size_t size);
  /*
   * \brief allocate shared memory that has been created.
   * \param size the size of the shared memory.
   * \return the address of the shared memory
   */
  void *Open(size_t size);

  /*
   * \brief check if the shared memory exist.
   * \param name the name of the shared memory.
   * \return a boolean value to indicate if the shared memory exists.
   */
  static bool Exist(const std::string &name);
};

// inline functions implementation after this
template<typename T>
inline bool Object::is_type() const {
  // use static field so query only happens once.
  static uint32_t type_id = Object::TypeKey2Index(T::_type_key);
  return type_id == this->type_index();
}

template<typename T>
inline bool Object::derived_from() const {
  // use static field so query only happens once.
  static uint32_t type_id = Object::TypeKey2Index(T::_type_key);
  return this->_DerivedFrom(type_id);
}

inline bool ObjectRef::defined() const {
  return obj_.get() != nullptr;
}

inline bool Object::_DerivedFrom(uint32_t tid) const {
  static uint32_t tindex = TypeKey2Index(Object::_type_key);
  return tid == tindex;
}

// this is slow, usually caller always hold the result in a static variable.
inline uint32_t Object::TypeKey2Index(const char* key) {
  TypeManager *t = TypeManager::Global();
  std::lock_guard<std::mutex> lock(t->mutex);
  std::string skey = key;
  auto it = t->key2index.find(skey);
  if (it != t->key2index.end()) {
    return it->second;
  }
  uint32_t tid = ++(t->type_counter);
  t->key2index[skey] = tid;
  t->index2key.push_back(skey);
  return tid;
}

inline const char* Object::TypeIndex2Key(uint32_t index) {
  TypeManager *t = TypeManager::Global();
  std::lock_guard<std::mutex> lock(t->mutex);
  CHECK_NE(index, 0);
  return t->index2key.at(index - 1).c_str();
}

inline bool ObjectRef::operator==(const ObjectRef& other) const {
  return obj_.get() == other.obj_.get();
}

inline bool ObjectRef::same_as(const ObjectRef& other) const {
  return obj_.get() == other.obj_.get();
}

inline bool ObjectRef::operator<(const ObjectRef& other) const {
  return obj_.get() < other.obj_.get();
}

inline bool ObjectRef::operator!=(const ObjectRef& other) const {
  return obj_.get() != other.obj_.get();
}

inline size_t ObjectRef::hash() const {
  return std::hash<Object*>()(obj_.get());
}

inline uint32_t ObjectRef::type_index() const {
  CHECK(obj_.get() != nullptr) << "null type";
  return get()->type_index();
}

inline const Object* ObjectRef::operator->() const {
  return obj_.get();
}

inline const Object* ObjectRef::get() const {
  return obj_.get();
}

template<typename T>
inline const T* ObjectRef::as() const {
  const Object* ptr = get();
  if (ptr && ptr->is_type<T>()) {
    return static_cast<const T*>(ptr);
  }
  return nullptr;
}

// ShareMem
inline SharedMemory::SharedMemory(const std::string &name) {
#ifndef _WIN32
  this->name = name;
  this->own = false;
  this->fd = -1;
  this->ptr = nullptr;
  this->size = 0;
#else
  std::cout << "[FATAL] Shared memory is not supported on Windows." << std::endl;
#endif  // _WIN32
}

inline SharedMemory::~SharedMemory() {
#ifndef _WIN32
  munmap(ptr, size);
  close(fd);
  if (own) {
    std::cout << "[INFO] remove " << name << " for shared memory" << std::endl;
    shm_unlink(name.c_str());
  }
#else
  std::cout << "[FATAL] Shared memory is not supported on Windows." << std::endl;
#endif  // _WIN32
}

inline void *SharedMemory::CreateNew(size_t size) {
#ifndef _WIN32
  this->own = true;

  int flag = O_RDWR|O_CREAT;
  fd = shm_open(name.c_str(), flag, S_IRUSR | S_IWUSR);
  CHECK_NE(fd, -1) << "fail to open " << name << ": " << strerror(errno);
  auto res = ftruncate(fd, size);
  CHECK_NE(res, -1)
      << "Failed to truncate the file. " << strerror(errno);
  ptr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  CHECK_NE(ptr, MAP_FAILED)
      << "Failed to map shared memory. mmap failed with error " << strerror(errno);
  return ptr;
#else
  std::cout << "[FATAL] Shared memory is not supported on Windows." << std::endl;
#endif  // _WIN32
}

inline void *SharedMemory::Open(size_t size) {
#ifndef _WIN32
  int flag = O_RDWR;
  fd = shm_open(name.c_str(), flag, S_IRUSR | S_IWUSR);
  CHECK_NE(fd, -1) << "fail to open " << name << ": " << strerror(errno);
  ptr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
  CHECK_NE(ptr, MAP_FAILED)
      << "Failed to map shared memory. mmap failed with error " << strerror(errno);
  return ptr;
#else
  std::cout << "[FATAL] Shared memory is not supported on Windows." << std::endl;
#endif  // _WIN32
}

inline bool SharedMemory::Exist(const std::string &name) {
#ifndef _WIN32
  int fd = shm_open(name.c_str(), O_RDONLY, S_IRUSR | S_IWUSR);
  if (fd >= 0) {
    close(fd);
    return true;
  } else {
    return false;
  }
#else
  std::cout << "[FATAL] Shared memory is not supported on Windows." << std::endl;
#endif  // _WIN32
}


#endif  // OBJECT_H_