#ifndef _HDL_LIST_H_
#define _HDL_LIST_H_

/* Author: Jakub Pola, jakub.pola@gmail.com
 * HDL LIST - Hashed double linked list.
 * This is double the linked list implementation with UNIQUE elements.
 * Uniqueness here is understood by the uniqueness of Elements keys.
 * Each element is a pair of (key, value) where the key is a string literal.
 */

#include <stdbool.h>
#include <stddef.h>


#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief RSHash Hash Function from Robert Sedgwicks Algorithms in C book
 * @param str - string literal to hash
 * @param len - lenght of the string (strlen(str)) //TODO:: maybe hide len inside?
 * @return hash value
 */
static inline unsigned int RSHash (const char* str, const int len)
{
    unsigned int b    = 378551;
    unsigned int a    = 63689;
    unsigned int hash = 0;
    unsigned int i    = 0;

    for (i = 0; i < len; str++, i++)
    {
        hash = hash * a + (*str);
        a = a * b;
    }

    return hash;
}

// hdl_element of the hdl_list;
typedef struct hdl_element
{
    struct hdl_element* prev;
    struct hdl_element* next;
    void* value;
    unsigned int hash;

} hdl_element;

// hdl_list structure;
typedef struct hdl_list
{
    size_t count;
    hdl_element* head;
    hdl_element* tail;

    // during destruction procedure they requrire one of following depends on what is the value
    // status = clReleaseKernel(kernel);               // Release kernel.
    // status = clReleaseProgram(program);             // Release the program object.
    // status = clReleaseMemObject(inputBuffer);       // Release mem object.
    // status = clReleaseCommandQueue(commandQueue);   // Release  Command queue.
    // status = clReleaseContext(context);             // Release context.
    bool clobject;


} hdl_list;

//type of function which will free the list elements;
typedef void (*free_func_t) (void*);

/**
 * @brief hdl_create Create hdl_list
 * @param cl_type Support for OpenCL structures
 * @return Pointer to newly created hdl_list
 */
hdl_list* hdl_create (bool clobject);

/**
 * @brief hdl_destroy Delete all elements from the list and free the list itself
 * @param list Address of pointer to hdl_list
 */
void hdl_destroy (hdl_list **list);

/**
 * @brief hdl_destroy_with_func Delete all elements from the list and free the
 * list itself using custom free_funct_t
 * @param list Adrees of pointer to hdl_list
 * @param free_func free function for element values
 */
void hdl_destroy_with_func (hdl_list ** list, free_func_t free_func);

/**
 * @brief hdl_size Return the size of hdl_list
 * @param list Pointer to hdl_list
 * @return Number of hdl_elements in the list
 */
size_t hdl_size (const hdl_list *list);

/**
 * @brief hdl_push_front Insert (key, value) at the beginning of the hdl_list
 * if the key does not exists
 * @param list Pointer to hdl_list
 * @param key Key of the hdl_element
 * @param value Value of the hdl_element
 */
void hdl_push_front (hdl_list *list, const char* key, const void* value);

/**
 * @brief hdl_push_back Insert (key, value) at the end of the hdl_list
 * if the key does not exists
 * @param list Pointer to hdl_list
 * @param key Key of the hdl_element
 * @param value Value of the hdl_element
 */
void hdl_push_back (hdl_list *list, const char* key, const void* value);

/**
 * @brief hdl_insert Convenient function, just add the hdl_element
 * if don't exists.
 * @param list Pointer to hdl_list
 * @param key Key of the hdl_element
 * @param value Value of the hdl_element
 */
void hdl_insert (hdl_list *list, const char* key, const void* value);

/**
 * @brief hdl_find_by_hash Find hdl_element by hash value
 * @param list Pointer to hdl_list
 * @param hash Hash value
 * @return pointer to hdl_element, NULL if not found.
 */
hdl_element* hdl_find_by_hash (const hdl_list *list, const unsigned int hash);

/**
 * @brief hdl_find_by_key Find hdl_element by key
 * @param list Pointer to hdl_list
 * @param hash Hash value
 * @return pointer to hdl_element, NULL if not found.
 */
hdl_element* hdl_find_by_key (const hdl_list *list, const char* key);

/**
 * @brief hdl_pop_front Pop the first hdl_element from the list, the hdl_list
 * loose control over this hdl_element. User is responsible for proper removal
 * @param list Pointer to hdl_list
 * @return Pointer to hdl_element or NULL if empty
 */
hdl_element* hdl_pop_front (hdl_list *list);

/**
 * @brief hdl_pop_back Pop the last hdl_element from the list, the hdl_list
 * loose control over this hdl_element. User is responsible for proper removal
 * @param list Pointer to hdl_list
 * @return Pointer to hdl_element or NULL if empty
 */
hdl_element* hdl_pop_back (hdl_list *list);

/**
 * @brief hdl_get_value_by_hash Get the list's hdl_element by it's hash code
 * We just want to read it's value nothig more, list is still in control
 * @param list Pointer to hdl_list
 * @param hash Hash code of the hdl_element
 * @return pointer to hdl_element or NULL if not found.
 */
const hdl_element* hdl_get_element_by_hash (const hdl_list *list,
                                            const unsigned int hash);

/**
 * @brief hdl_get_element_by_key Get the list's hdl_element by it's key
 * We just want to read it's value nothig more, list is still in control
 * @param list Pointer to hd_list
 * @param hash Hash code of the hdl_element
 * @return pointer to hdl_element or NULL if not found.
 */
const hdl_element* hdl_get_element_by_key (const hdl_list *list,
                                           const char* key);

/**
 * @brief hdl_get_value_by_key Get value of the hdl_element indicated by key
 * @param list Pointer to hd_list
 * @param key Key code of the hdl_element
 * @return pointer to hdl_element value or NULL if not found.
 */
const void* hdl_get_value_by_key(const hdl_list *list, const char* key);

/**
 * @brief hdl_get_value_by_hash Get value of the hdl_element indicated
 * by hash code
 * @param list Pointer to hd_list
 * @param key Key code of the hdl_element
 * @return pointer to hdl_element value or NULL if not found.
 */
const void* hdl_get_value_by_hash(const hdl_list *list,
                                  const unsigned int hash);


/**
 * @brief hdl_delete_element Delete e from the list, frees resources at the end
 * @param list Pointer to hdl_list
 * @param e Pointer to hdl_element which needs to be erased
 */
void hdl_delete_element (hdl_list *list, hdl_element **e);

void hdl_delete_element_with_func (hdl_list *list, hdl_element **e,
                                    free_func_t free_func);

/**
 * @brief hdl_delete_element_by_hash Delete e from the list by its hash code
 * @param list Pointer to hdl_list
 * @param hash Hash code of element
 */
void hdl_delete_element_by_hash(hdl_list *list, const unsigned int hash);

void hdl_delete_element_by_hash_with_func (hdl_list *list, const unsigned int hash, free_func_t free_func);

/**
 * @brief hdl_delete_element_by_key Delete e from the list by its key value
 * @param list Pointer to hdl_list
 * @param key Key of the element
 */
void hdl_delete_element_by_key (hdl_list* list, const char* key);

void hdl_delete_element_by_key_with_func (hdl_list* list, const char* key, free_func_t free_func);

/**
 * @brief hdl_remove_element Removes the element from the list.
 * It does not free allocated resources. List loose control over this element.
 * User is responsible for proper resource management.
 * @param list Pointer to hdl_list
 * @param e pointer to hdl_element to be removed
 * @return pointer to released hdl_element (the same as e)
 */
hdl_element* hdl_remove_element (hdl_list *list, hdl_element* e);

/**
 * @brief hdl_remove_by_hash Removes the element from the list.
 * It does not free allocated resources. List loose control over this element.
 * User is responsible for proper resource management.
 * @param list Pointer to hdl_list
 * @param hash Hash code of the element
 * @return Pointer to released hdl_element inidcated by hash. Null if not found.
 */
hdl_element* hdl_remove_by_hash (hdl_list *list, const unsigned int hash);

/**
 * @brief hdl_remove_by_key Removes the element from the list.
 * It does not free allocated resources. List loose control over this element.
 * User is responsible for proper resource management.
 * @param list Pointer to hdl_list
 * @param key key of the element
 * @return Pointer to released hdl_element inidcated by hash. Null if not found.
 */
hdl_element* hdl_remove_by_key (hdl_list *list, const char* key);

/**
 * @brief hdl_print Debug function, print all elements from hdl_list
 * @param list pointer to hdl_list
 */
void hdl_print(const hdl_list *list);

#ifdef __cplusplus
}      /* extern "C" { */
#endif

#endif //__HDL_LIST_H__
