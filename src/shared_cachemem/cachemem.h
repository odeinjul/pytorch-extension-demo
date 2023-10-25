#ifndef CACHEMEM_H_
#define CACHEMEM_H_

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "./object.h"


typedef int32_t IdType;
typedef int64_t CacheIdxType;
const int CACHE_LINE_SIZE = 8;

template<class IdType, int Size>
// idx: index of cacheline
// id: index of embedding or gradients
// size: size of cacheline
class cache_line{
    char data[(sizeof(IdType) + sizeof(int32_t)) * Size];
    int access_table[Size];

    public:
        cache_line(){
            //init cache line
            for(int i = 0; i < Size; i++){
                set(i, -1, -1);
                access_table[i] = -1;
            }
        }

        void set(int idx, IdType id, int32_t loc){
            reinterpret_cast<IdType *>(data)[idx] = id; // set index of embedding
            reinterpret_cast<int32_t *>(data + sizeof(IdType) * Size)[idx] = loc; // set loc in cache
            access_table[idx] = 0; // set access number to 0
        }

        IdType get_id(int idx) const{
            // return a index of embedding according to given index in cache line
            return reinterpret_cast<const IdType *>(data)[idx];
        }

        int32_t get_loc(int idx) const{
            // return a loc of embedding according to given index in cache line
            return reinterpret_cast<const int32_t *>(data + sizeof(IdType) * Size)[idx];
        }

        bool is_init(int idx) const{
            // if a returned index of embedding is not equal to -1 -> is init
            return this->get_id(idx) != -1;
        }

        int get_valid_entries() const{
            // return number of valid entries in this cache line 
            int valid = 0;
            for(int i = 0; i < Size; i++){
                valid += is_init(i);
            }
            return valid;
        }

        int find(IdType id) const{
            // find a index of embedding equal to id in this cache line
            // if so, return the index of it 
            // if not, return -1
            for(int i = 0; i < Size; i++){
                if(get_id(i) == id)
                    return i;
            }
            return -1;
        }

        int find_empty_entry() const{
            // find an entry in this cache line which is not valid (or id == -1)
            for(int i = 0; i < Size; i++){
                if(!is_init(i))
                    return i;
            }
            return -1;
        }

        void access_loc(int idx){
            // increase access number when accessing
            for(int i = 0; i < Size; i++){
                access_table[idx] += 1;
            }
            access_table[idx] = 0;
        }

        int find_lru(){
            int idx = 0;
            int tmp = access_table[idx];
            for(int i = 0; i < Size; i++){
                if(access_table[i] > tmp && access_table[i] != 0){
                    idx = i;
                    tmp = access_table[i];
                }
            }
            if(tmp == 0) return -1; // this could happen when all enrties are new data
            return idx;
        }
};


class SACacheIndex : public Object{
    //cache_size: total entries of cache
    //index_size: number of cache lines
    //

    using CacheLineType = cache_line<IdType, CACHE_LINE_SIZE>;
    CacheLineType *index;
    std::shared_ptr<SharedMemory> shared_mem = nullptr;
    size_t index_size;
    CacheIdxType cache_size;

    const cache_line<IdType, CACHE_LINE_SIZE> &get_line(IdType id) const{
        return index[id % this->index_size];
    }

    cache_line<IdType, CACHE_LINE_SIZE> &get_line(IdType id){
        return index[id % this->index_size];
    }

    void init_cache(){
        // init cache to (-1, locs++)
        CacheIdxType cache_idx = 0;
        for(size_t i = 0; i < this->index_size; i++){
            for(int j = 0; j < CACHE_LINE_SIZE; j++){
                index[i].set(j, -1, cache_idx++);
            }
        }
        this->cache_size = cache_idx;
    }

    public:
        explicit SACacheIndex(size_t cache_size){
            this->index_size = cache_size / CACHE_LINE_SIZE;
            this-> index = new CacheLineType[this->index_size];
            this->init_cache();
        }

        SACacheIndex(size_t cache_size, const std::string &name, bool create = false){
            this->shared_mem = std::make_shared<SharedMemory>(name);
            this->index_size = cache_size / CACHE_LINE_SIZE;
            size_t shared_mem_size = sizeof(this->cache_size);
            shared_mem_size += sizeof(CacheLineType) * this->index_size;
            char* mem;
            if(create)
                mem = static_cast<char *>(this->shared_mem->CreateNew(shared_mem_size));
            else
                mem = static_cast<char *>(this->shared_mem->Open(shared_mem_size));
            auto shared_cache_size = reinterpret_cast<size_t *>(mem);
            mem += sizeof(this->cache_size);
            this->index = reinterpret_cast<CacheLineType *>(mem);
            if (create) {
                for (size_t i = 0 ; i < this->index_size ; i++)
                    new (mem + i)CacheLineType();
                this->init_cache();
                *shared_cache_size = this->cache_size;
            }
            else this->cache_size = *shared_cache_size;
        }

        virtual ~SACacheIndex(){
           if (!this->shared_mem){
               delete []this->index;
           } 
        }

        template<class LookupType>
        void add(const LookupType *ids, size_t len, CacheIdxType *locs){
            int lru_cnt = 0;
            for (size_t i = 0; i < len; i++){
                cache_line<IdType, CACHE_LINE_SIZE> &line = get_line(ids[i]);
                int idx = line.find_empty_entry();
                if (idx >= 0){
                    CacheIdxType cache_loc = line.get_loc(idx);
                    line.set(idx, ids[i], cache_loc);
                    locs[i] = cache_loc;
                }
                else if(idx == -1){
                    // using LRU
                    int idx = line.find_lru();
                    if (idx == -1){
                        locs[i] = -1;
                        lru_cnt++;
                    }
                    else{
                        CacheIdxType cache_loc = line.get_loc(idx);
                        line.set(idx, ids[i], cache_loc);
                        locs[i] = cache_loc;
                    }
                }
                else{
                    locs[i] = -1;
                }
            }
        }

        int64_t get_cache_size() const{
            return this->cache_size;
        }

        int64_t get_valid_entries() const{
            size_t valid = 0;
            for (size_t i = 0; i < this->index_size; i++)
                valid += index[i].get_valid_entries();
            return valid;
        }

        size_t get_capacity() const{
            return this->index_size * CACHE_LINE_SIZE;
        }

        size_t get_space() const{
            return this->index_size * sizeof(cache_line<IdType, CACHE_LINE_SIZE>);
        }
        
        template<class LookupType>
        void lookup(const LookupType *ids, int64_t len, CacheIdxType *locs,
                    LookupType *return_ids){
#pragma omp parallel for
            for(int64_t i = 0; i < len; i++){
                cache_line<IdType, CACHE_LINE_SIZE> &line = get_line(ids[i]);
                int entry_idx = line.find(ids[i]);
                if(entry_idx == -1){
                    // If the id dosn't exist,
                    // the location is set to the end of the cache,
                    // the corresponding id is set to -1.
                    locs[i] = cache_size;
                    return_ids[i] = -1;
                }
                else{
                    locs[i] = line.get_loc(entry_idx);
                    line.access_loc(entry_idx);
                    return_ids[i] = ids[i];
                }
            }
        }

        template<class LookupType>
        void reset(const LookupType *ids, int64_t len){
#pragma omp parallel for
            for(int64_t i = 0; i < len; i++){
                cache_line<IdType, CACHE_LINE_SIZE> &line = get_line(ids[i]);
                int entry_idx = line.find(ids[i]);
                if(entry_idx == -1);
                else{
                    CacheIdxType cache_loc = line.get_loc(entry_idx);
                    line.set(entry_idx, -1, cache_loc);
                }
            }
        }

        template<class LookupType>
        void getall(LookupType *ids, CacheIdxType *locs, size_t len){
            if (len != this->get_valid_entries()){
                printf("len: %d ve: %d\n", len, this->get_valid_entries());
            }
            size_t ids_idx = 0;
            for (size_t i = 0; i < this->index_size; i++){
                for(size_t j = 0; j < CACHE_LINE_SIZE; j++){
                    if(index[i].get_id(j) != -1){
                        ids[ids_idx] = index[i].get_id(j);
                        locs[ids_idx] = index[i].get_loc(j);
                        ids_idx++;
                        if(ids_idx == len) return;
                    }
                }
            }
        }

        static constexpr const char* _type_key = "cache.SACache";        
    
		const char* type_key() const final {                                   
        	return SACacheIndex::_type_key;                                          
    	}                                                
                          
    	uint32_t type_index() const final {                                    
        	static uint32_t tidx = TypeKey2Index(SACacheIndex::_type_key);           
        	return tidx;                                                         
    	}                                                 

    	bool _DerivedFrom(uint32_t tid) const final {                          
        	static uint32_t tidx = TypeKey2Index(SACacheIndex::_type_key);           
        	if (tidx == tid) return true;                                        
        	return Object::_DerivedFrom(tid);                                    
    	}

    //TODO: where is VisitAttrs()?
};


class CacheRef : public ObjectRef {                                 
    public:                                                                                  
    CacheRef() {}   
                                                                    
    explicit CacheRef(std::shared_ptr<Object> obj): ObjectRef(obj) {}   
    
    const SACacheIndex* operator->() const {                                          
        return static_cast<const SACacheIndex*>(obj_.get());                            
    }                                                                               

    SACacheIndex* operator->() {                                                      
        return static_cast<SACacheIndex*>(obj_.get());                                  
    }                                                                               

    std::shared_ptr<SACacheIndex> sptr() const {                                      
        return CHECK_NOTNULL(std::dynamic_pointer_cast<SACacheIndex>(obj_));            
    }                                                                               

    operator bool() const { return this->defined(); }                               

    using ContainerType = SACacheIndex;

};

CacheRef SharedMemCacheCreate (size_t cache_size, std::string name, bool create);
#endif //#CACHEMEM_H_