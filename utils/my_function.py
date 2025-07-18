# def square_list(mylist, result, square_sum):
#     """
#     function to square a given list
#     """
#     # append squares of mylist to result array
#     for idx, num in enumerate(mylist):
#         result[idx] = num * num
#
#     # square_sum value
#     square_sum.value = sum(result)
#
#     # print result Array
#     print("Result(in process p1): {}".format(result[:]))
#
#     # print square_sum Value
#     print("Sum of squares(in process p1): {}".format(square_sum.value))


def square_list(mylist, q):
    """
    function to square a given list
    """
    # append squares of mylist to queue
    for num in mylist:
        q.put(num * num)


def print_queue(q):
    """
    function to print queue elements
    """
    print("Queue elements:")
    while not q.empty():
        print(q.get())
    print("Queue is now empty!")
