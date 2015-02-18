/**
    lets try to implement a hash table with  keys  of type const char*
    which return const char* value;
*/

#include "hdl_list.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


hdl_list* hdl_create (bool clobject)
{
    hdl_list* list = malloc (sizeof (hdl_list));

    if (list == NULL) return NULL;

    list->head = NULL;
    list->tail = NULL;
    list->count = 0;
    list->clobject = clobject;
    return list;
}

size_t hdl_size (const hdl_list *list)
{
    return list->count;
}

/* FIND FUNCTION SECTION */

hdl_element* hdl_find_by_hash (const hdl_list *list, const unsigned int hash)
{
    hdl_element *p = list->head;

    //TODO: Wartowinik? chyba nie potrzebny bo to będzie mała lista
    int index = 0;
    while (p && p->hash != hash)
    {
        p = p->next;
    }

    return p;
}

hdl_element* hdl_find_by_key (const hdl_list *list, const char* key)
{
    unsigned int hash = RSHash(key, strlen(key));

    return hdl_find_by_hash(list, hash);
}


/* INSERT FUNCTION SECTION */


// TODO: what should be value void*, const void*??
// TODO: Error handling ?? Jaki sposób;
// Jeżeli wkładam coś do listy to odpowiedzialność spada na listę aby
// element dobrze usunąć
void hdl_push_front(hdl_list *list, const char* key, const void* value)
{
    unsigned int hash = RSHash (key, strlen(key));

    // Chcemy mieć unikalne elementy w liście;
    hdl_element* e = hdl_find_by_hash(list, hash);

    if (e == NULL )  // nie znaleziono elementu o kluczu = key
    {
        e = malloc  (sizeof (hdl_element));     // nie rzutować w C
        e->value = (void*)value;           // przenoszę adres do listy
        e->hash = hash;             // dodajemy identyfikator
        e->prev = NULL;             // pierwszy element w liście nie posiada poprzednika;
        e->next = list->head;     // następny element po pierwszym to głowa
        list->head = e;                // głowa wskazuje na pierwszy element
        list->count++;                // zwiekszamy ilość elementów w liście

        // komunikacja z ogonem
        // jeżeli e posiada następnik np c dodanego elementu to musimy go podłączyć
        // e->next oznacza ze idziemy head->e->c, gdzie c = e->next;
        // gdy wywołujemy prev na elemencie to zbliżamy się do głowy! stąd e->next->prev = e;
        if(e->next)
            e->next->prev = e;
        else
            list->tail = e;     // w przeciwnym razie e jest jedynym elementem w liscie; head i tail wskazuje na e;
    }
#ifndef NDEBUG
    else
    {
        fprintf(stderr, "Key (%s, %u) already exists\n Element will is not added\n", key, hash);
    }
#endif
}

//TODO: Te same pytania co hdl_push_front
void hdl_push_back(hdl_list *list, const char* key, const void* value)
{
    unsigned int hash = RSHash (key, strlen(key));

    hdl_element* e = hdl_find_by_hash(list, hash);

    if (e == NULL )
    {
        e = malloc (sizeof (hdl_element));
        e->value = (void*)value;
        e->hash = hash;
        e->next = NULL;
        e->prev = list->tail;
        list->tail = e;
        list->count++;

        //komunikacja z głową listy symetrycznie do push_front;
        if (e->prev)
            e->prev->next = e;
        else
            list->head = e;

    }
#ifndef NDEBUG
    else
    {
        fprintf(stderr, "Key (%s, %u) already exists\n Element is not added\n",
                key, hash);
    }
#endif
}

void hdl_insert (hdl_list *list, const char* key, const void* value)
{
    hdl_push_back(list, key, value);
}

/** REMOVE / DELETE FUNCTION SECTION */

// helper function used in remove or delete, just properly reorganize
// the list when removing e
static void __hdl_remove_element(hdl_list *list, hdl_element* e)
{
    if (e == NULL) return;      // node not found;

    list->count--;

    // odpowiednie poloczenie elementow w okółe w 2 krokach
    // 1. jesli istniej poprzednik e to w jego polu next
    //    umieszczamy to co bylo w e->prev
    //
    //      head --- c ---  e --- f
    //      e - current element
    //      c := e->prev;
    //      f := e->next;
    //      c->next musi wskazywać na adres f;
    //      f->prev musi wskazywać na adres c;
    if (e->prev) e->prev->next = e->next;
    else list->head = e->next;    // jesli nie ma prev to e był pierwszym
    // elementem i teraz musimy wskazać na f;

    // 2. to samo z ogonem
    if (e->next) e->next->prev = e->prev;
    else list->tail = e->prev;
}

// helper function used to properly free allocated resources
static void __hdl_free_element(hdl_list *list , hdl_element** e,
                               free_func_t free_func)
{
    if(*e)
    {

        if(list->clobject) free_func((*e)->value);
        free(*e);
    }

    *e = NULL;

}

//delete element from the list, calls free(e) at the end
void hdl_delete_element (hdl_list *list, hdl_element** e)
{
    __hdl_remove_element(list, *e);
    __hdl_free_element(list, e, &free);
}

void hdl_delete_element_with_func (hdl_list *list, hdl_element **e,
                                   free_func_t free_func)
{
    __hdl_remove_element(list, *e);
    __hdl_free_element(list, e, free_func);
}

void hdl_delete_element_by_hash(hdl_list *list, const unsigned int hash)
{
    hdl_element* e = hdl_find_by_hash(list, hash);
    hdl_delete_element(list, &e);
}

void hdl_delete_element_by_hash_with_func(hdl_list *list, const unsigned int hash,
                                          free_func_t free_func)
{
    hdl_element* e = hdl_find_by_hash(list, hash);
    hdl_delete_element_with_func (list, &e, free_func);
}


void hdl_delete_element_by_key (hdl_list* list, const char* key)
{
    unsigned int hash = RSHash(key, strlen(key));
    hdl_delete_element_by_hash(list, hash);
}

void hdl_delete_element_by_key_with_func (hdl_list *list, const char *key,
                                          free_func_t free_func)
{
    unsigned int hash = RSHash(key, strlen(key));
    hdl_delete_element_by_hash_with_func (list, hash, free_func);
}


//remove element from the list, does not testroy it
hdl_element* hdl_remove_element (hdl_list *list, hdl_element* e)
{
    __hdl_remove_element(list, e);
    return e;
}

hdl_element* hdl_remove_by_hash (hdl_list *list, const unsigned int hash)
{
    hdl_element* e = hdl_find_by_hash(list, hash);
    return hdl_remove_element(list, e);
}

hdl_element* hdl_remove_by_key (hdl_list *list, const char* key)
{
    unsigned int hash = RSHash(key, strlen(key));
    return hdl_remove_by_hash(list, hash);
}

// TODO: Can we make it better?
void hdl_destroy(hdl_list **list)
{
    if (!(*list)) return;

    hdl_element* e = (*list)->head;
    hdl_element* tmp = NULL;
    while (e)
    {
        tmp = e->next;
        hdl_delete_element(*list, &e);
        e = tmp;
    }
    //    for (e = (*list)->head; e != NULL; e = e->next)
    //    {
    //        hdl_delete_element(*list, &e);
    //    }
    free(*list);
    *list = NULL;
}

void hdl_destroy_with_func (hdl_list **list, free_func_t free_func)
{
    if (!(*list)) return;

    hdl_element* e = (*list)->head;
    hdl_element* tmp = NULL;
    while (e)
    {
        tmp = e->next;
        hdl_delete_element_with_func (*list, &e, free_func);
        e = tmp;
    }

    free(*list);
    *list = NULL;

}

// jeżeli coś ściągamy z kolejki to kolejka traci nad nim kontrolę
// my musimy zadbać aby odpowienio usunąć go z pamięci
hdl_element* hdl_pop_front (hdl_list *list)
{
    if(list->count)
        return hdl_remove_element(list, list->head);

    return NULL;
}

hdl_element* hdl_pop_back (hdl_list *list)
{
    if(list->count)
        return hdl_remove_element(list, list->tail);

    return NULL;
}


//we just want to read data from hdl_element
const hdl_element* hdl_get_element_by_hash (const hdl_list *list, const unsigned int hash)
{
    return hdl_find_by_hash(list, hash);
}

const hdl_element* hdl_get_element_by_key (const hdl_list *list, const char* key)
{
    const unsigned int hash = RSHash (key, strlen(key));
    return hdl_get_element_by_hash (list, hash);
}

const void* hdl_get_value_by_key(const hdl_list *list, const char* key)
{
    const hdl_element* e = hdl_get_element_by_key(list, key);
    if(e)
        return e->value;
    return NULL;
}

const void* hdl_get_value_by_hash(const hdl_list *list, const unsigned int hash)
{
    const hdl_element* e = hdl_get_element_by_hash(list, hash);
    if(e)
        return e->value;
    return NULL;
}

// diagnostic function
void hdl_print (const hdl_list *list)
{
    hdl_element* e = list->head;

    while (e)
    {
        printf("(%u, %s)\n", e->hash, (const char*)e->value);
        e = e->next;
    }
}
