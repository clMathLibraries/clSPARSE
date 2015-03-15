#include "../library/internal/hdl_list.h"
#include "resources/hdl_test_resources.h"
#include <gtest/gtest.h>

TEST (hdl_list, create)
{
    hdl_list* L = NULL;
    L = hdl_create(false);

    EXPECT_TRUE(L != NULL);
    EXPECT_EQ(L->clobject, false);

    hdl_insert(L, k1, source1);
    hdl_insert(L, k2, source2);
    hdl_insert(L, k3, source3);

    EXPECT_EQ(3, hdl_size(L));

    hdl_destroy(&L);
    EXPECT_TRUE(L == NULL);
}

TEST (hdl_list, destroy)
{
    hdl_list* L = NULL;
    L = hdl_create(false);

    EXPECT_TRUE(L != NULL);

    hdl_destroy(&L);
    hdl_destroy(&L);

    EXPECT_TRUE(L == NULL);
}

TEST (hdl_list, add)
{
    hdl_list* L = hdl_create(false);
    EXPECT_TRUE(L != NULL);

    hdl_push_back(L, k1, source1);
    EXPECT_EQ(1, hdl_size(L));

    hdl_push_front(L, k2, source1);
    EXPECT_EQ(2, hdl_size(L));

    hdl_insert(L, k3, source3);
    EXPECT_EQ(3, hdl_size(L));

    hdl_destroy(&L);
    EXPECT_TRUE(L == NULL);
}

TEST (hdl_list, add_duplicated)
{
    hdl_list* L = hdl_create(false);
    EXPECT_TRUE(L != NULL);

    hdl_push_back(L, k1, source1);
    EXPECT_EQ(1, hdl_size(L));

    hdl_push_front(L, k2, source1);
    EXPECT_EQ(2, hdl_size(L));

    hdl_insert(L, k2, source2);
    EXPECT_EQ(2, hdl_size(L));

    hdl_destroy(&L);
    EXPECT_TRUE(L == NULL);

}

TEST (hdl_list, pop_back)
{
    hdl_list* L = hdl_create (false);

    EXPECT_TRUE (L != NULL);

    hdl_push_front (L, k1, source1); //push front causes that this is the last;
    hdl_push_front (L, k2, source2);
    hdl_push_front (L, k3, source3);

    EXPECT_EQ(3, hdl_size(L));

    hdl_element* e = hdl_pop_back(L);

    EXPECT_EQ(2, hdl_size(L));

    EXPECT_STREQ(source1, (const char*)e->value);

    // remember to take care of this element when poped
    // list looses control over it after pop
    free(e);

    hdl_destroy(&L);
    EXPECT_TRUE(L == NULL);
}

TEST (hdl_list, pop_front)
{

    hdl_list* L = hdl_create (false);

    EXPECT_TRUE (L != NULL);

    hdl_push_front (L, k1, source1); //push front causes that this is the last
    hdl_push_front (L, k2, source2);
    hdl_push_front (L, k3, source3);

    EXPECT_EQ(3, hdl_size(L));

    hdl_element* e = hdl_pop_front(L);

    EXPECT_EQ(2, hdl_size(L));

    EXPECT_STREQ(source3, (const char*)e->value);

    // remember to take care of this element when poped
    // list looses control over it after pop
    free(e);

    hdl_destroy(&L);
    EXPECT_TRUE(L == NULL);
}

TEST (hdl_list, find_by_hash)
{

    hdl_list* L = hdl_create (false);

    EXPECT_TRUE(L != NULL);

    hdl_insert (L, k1, source1);
    hdl_insert (L, k2, source2);
    hdl_insert (L, k3, source3);
    hdl_insert (L, k4, source4);

    EXPECT_EQ(4, hdl_size(L));

    //find k3 and cmpare with source3
    unsigned int hash3 = RSHash(k3, strlen(k3));
    hdl_element *e1 = hdl_find_by_hash(L, hash3);

    EXPECT_TRUE (e1 != NULL);
    EXPECT_STREQ(source3, (const char*)e1->value);

    hdl_destroy (&L);
    EXPECT_TRUE(L == NULL);
}

TEST (hdl_list, find_by_hash_notFound)
{

    hdl_list* L = hdl_create (false);

    EXPECT_TRUE(L != NULL);

    hdl_insert (L, k1, source1);
    hdl_insert (L, k2, source2);
    hdl_insert (L, k3, source3);
    hdl_insert (L, k4, source4);

    EXPECT_EQ(4, hdl_size(L));

    //find k6 which is not currently in the list
    unsigned int hash6 = RSHash(k6, strlen(k6));
    hdl_element *e = hdl_find_by_hash(L, hash6);

    EXPECT_TRUE (e == NULL);

    hdl_destroy (&L);
    EXPECT_TRUE(L == NULL);
}

TEST(hdl_list, find_by_key)
{
    hdl_list* L = hdl_create (false);

    EXPECT_TRUE(L != NULL);

    hdl_insert (L, k1, source1);
    hdl_insert (L, k2, source2);
    hdl_insert (L, k3, source3);
    hdl_insert (L, k4, source4);

    EXPECT_EQ(4, hdl_size(L));

    hdl_element *e = hdl_find_by_key(L, k3);

    EXPECT_TRUE (e != NULL);
    EXPECT_STREQ(source3, (const char*)e->value);

    hdl_destroy (&L);
    EXPECT_TRUE(L == NULL);

}

TEST(hdl_list, get_by_key)
{
    hdl_list* L = hdl_create (false);

    EXPECT_TRUE (L != NULL);

    hdl_insert (L, k1, source1);
    hdl_insert (L, k2, source2);
    hdl_insert (L, k3, source3);
    hdl_insert (L, k4, source4);

    //find k3 and cmpare with source3
    const hdl_element *e1 = hdl_get_element_by_key(L, k4);

    EXPECT_TRUE (e1 != NULL);
    EXPECT_STREQ (source4, (const char*)e1->value);

    //try to find sth not present
    const hdl_element *e2 = hdl_get_element_by_key(L, k6);
    EXPECT_TRUE (e2 == NULL);

    hdl_destroy (&L);
    EXPECT_TRUE (L == NULL);
}

TEST(hdl_list, get_by_hash)
{
    hdl_list* L = NULL;
    L = hdl_create (false);

    EXPECT_TRUE (L != NULL);

    hdl_insert (L, k1, source1);
    hdl_insert (L, k2, source2);
    hdl_insert (L, k3, source3);
    hdl_insert (L, k4, source4);

    //find k3 and cmpare with source3
    const unsigned int hash4 = RSHash(k4, strlen(k4));
    const hdl_element *e1 = hdl_get_element_by_hash(L, hash4);

    EXPECT_TRUE (e1 != NULL);
    EXPECT_STREQ (source4, (const char*)e1->value);

    //try to find sth not present
    const unsigned int hash6 = RSHash(k6, strlen(k6));
    const hdl_element *e2 = hdl_get_element_by_hash(L, hash6);
    EXPECT_TRUE (e2 == NULL);

    hdl_destroy (&L);
    EXPECT_TRUE (L == NULL);
}

TEST (hdl_list, get_value_by_key)
{
    hdl_list* L = NULL;
    L = hdl_create (false);

    EXPECT_TRUE (L != NULL);

    hdl_insert (L, k1, source1);
    hdl_insert (L, k2, source2);
    hdl_insert (L, k3, source3);
    hdl_insert (L, k4, source4);

    //find k3 and cmpare with source3
    const char* e1 = (const char*) hdl_get_value_by_key(L, k4);

    EXPECT_TRUE (e1 != NULL);
    EXPECT_STREQ (source4, e1);

    //try to find sth not present
    const char* e2 = (const char*) hdl_get_value_by_key(L, k6);
    EXPECT_TRUE (e2 == NULL);

    hdl_destroy (&L);
    EXPECT_TRUE (L == NULL);
}

TEST (hdl_list, get_value_by_hash)
{
    hdl_list* L = NULL;
    L = hdl_create (false);

    EXPECT_TRUE (L != NULL);

    hdl_insert (L, k1, source1);
    hdl_insert (L, k2, source2);
    hdl_insert (L, k3, source3);
    hdl_insert (L, k4, source4);

    //find k3 and cmpare with source3
    const unsigned int hash4 = RSHash(k4, strlen(k4));
    const char *e1 = (const char*) hdl_get_value_by_hash(L, hash4);

    EXPECT_TRUE (e1 != NULL);
    EXPECT_STREQ (source4, e1);

    //try to find sth not present
    const unsigned int hash6 = RSHash(k6, strlen(k6));
    const char *e2 = (const char*) hdl_get_value_by_hash(L, hash6);
    EXPECT_TRUE (e2 == NULL);

    hdl_destroy (&L);
    EXPECT_TRUE (L == NULL);
}

TEST (hdl_list, remove_by_key)
{
    hdl_list* L = NULL;
    L = hdl_create (false);

    EXPECT_TRUE (L != NULL);

    hdl_push_front (L, k1, source1); //push front causes that this is the last
    hdl_push_front (L, k2, source2);
    hdl_push_front (L, k3, source3);

    EXPECT_EQ (3, hdl_size(L));

    hdl_element *e = hdl_remove_by_key(L, k2); //remove element existing in the list
    EXPECT_EQ (2, hdl_size(L));
    EXPECT_STREQ(source2, (const char*)e->value);

    hdl_element *e1 = hdl_remove_by_key(L, k7);//remve non existing element from the list
    EXPECT_TRUE (e1 == NULL);
    EXPECT_EQ (2, hdl_size(L));

    free(e);
    e = NULL; //remember to set it to null! avoids problems

    hdl_destroy(&L);
    EXPECT_TRUE (L == NULL);
}

TEST (hdl_list, remove_by_hash)
{
    hdl_list* L = NULL;
    L = hdl_create (false);

    EXPECT_TRUE (L != NULL);

    hdl_push_front (L, k1, source1); //push front causes that this is the last
    hdl_push_front (L, k2, source2);
    hdl_push_front (L, k3, source3);

    EXPECT_EQ (3, hdl_size(L));

    unsigned int h1 = RSHash(k2, strlen(k2));
    unsigned int h2 = RSHash(k7, strlen(k7));

    hdl_element *e = hdl_remove_by_hash(L, h1); //remove element existing in the list
    EXPECT_EQ (2, hdl_size(L));
    EXPECT_STREQ(source2, (const char*)e->value);

    hdl_element *e1 = hdl_remove_by_hash(L, h2);//remve non existing element from the list
    EXPECT_TRUE (e1 == NULL);
    EXPECT_EQ (2, hdl_size(L));

    free(e);
    e = NULL;

    hdl_destroy(&L);
    EXPECT_TRUE (L == NULL);
}

TEST (hdl_list, delete_by_key)
{
    hdl_list* L = NULL;
    L = hdl_create (false);

    EXPECT_TRUE (L != NULL);

    hdl_push_front (L, k1, source1);
    hdl_push_front (L, k2, source2);
    hdl_push_front (L, k3, source3);

    EXPECT_EQ(3, hdl_size(L));

    hdl_delete_element_by_key(L, k2);
    EXPECT_EQ(2, hdl_size(L));
    //we shall not find k2 anymore in the list;
    EXPECT_TRUE(hdl_find_by_key(L, k2) == NULL);

    hdl_destroy(&L);
    EXPECT_TRUE (L == NULL);
}

TEST (hdl_list, delete_by_hash)
{
    hdl_list* L = NULL;
    L = hdl_create (false);

    EXPECT_TRUE (L != NULL);

    hdl_push_front (L, k1, source1);
    hdl_push_front (L, k2, source2);
    hdl_push_front (L, k3, source3);

    EXPECT_EQ(3, hdl_size(L));

    unsigned int h = RSHash(k2, strlen(k2));
    hdl_delete_element_by_hash(L, h);
    EXPECT_EQ(2, hdl_size(L));
    //we shall not find k2 anymore in the list;
    EXPECT_TRUE (hdl_find_by_hash(L, h) == NULL);

    hdl_destroy(&L);
    EXPECT_TRUE (L == NULL);
}

TEST (hdl_list, list_delete_element)
{
    hdl_list* L = NULL;
    L = hdl_create (false);

    EXPECT_TRUE (L != NULL);

    hdl_push_front (L, k1, source1);
    hdl_push_front (L, k2, source2);
    hdl_push_front (L, k3, source3);

    EXPECT_EQ(3, hdl_size(L));

    hdl_element *e = hdl_find_by_key(L, k2);

    hdl_delete_element(L, &e);

    EXPECT_TRUE(e == NULL);
    EXPECT_EQ(2, hdl_size(L));
    EXPECT_TRUE(hdl_find_by_key(L, k2) == NULL);

    hdl_destroy(&L);
    EXPECT_TRUE (L == NULL);
}


int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
