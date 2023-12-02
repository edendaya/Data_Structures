#username - yanivmualem
#id1      - 209127612
#name1    - Yanivm Mualem
#id2      - 315285759
#name2    - Eden Daya

import random
"""A class representing a node in an AVL tree"""

class AVLNode(object):
    """Constructor, you are allowed to add more fields.

    @type value: str
    @type is_virtual: boolean optional
    @param value: data of your node
    """

    def __init__(self, value, is_virtual=False):
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.height = -1 if is_virtual else 0
        self.size = 0 if is_virtual else 1
        self.is_virtual = is_virtual;


    """calculating balance factor of self and returns it
    @rtype: int
    @returns: balance factor of self
    """

    def calcBalanceFactor(self):
        return 0 if self.is_virtual else self.getLeft().height - self.getRight().height

    """finding the root of the node and returns it
    @rtype: AVLNode
    @returns: the root tree of the node
    """

    def findingRoot(self):
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    """adding left or right leaf to the node so that each node will have right child and left child
    @rtype: Void
    @returns: Void
    """

    def addingVirtualNodes(self):
        if self.left is None:
            self.setLeft(AVLNode(None, True))
        if self.right is None:
            self.setRight(AVLNode(None, True))

    """check if node is right child of self
    @rtype: boolean
    @returns: true if node.getRight == self else false
    """
    def rightChildCheck(self, node):
        if node is None:
            return False
        return node.getRight() == self

    """check if node is right child of self
    @rtype: boolean
    @returns: true if node.getRight == self else false
    """
    def leftChildCheck(self, node):
        if node is None:
            return False
        return node.getLeft() == self

    """returns the left child
    @rtype: AVLNode
    @returns: the left child of self, None if there is no left child
    """
    def getLeft(self):
        if self.left is None:
            self.left = AVLNode(None,True)
        return self.left

    """returns the right child
    @rtype: AVLNode
    @returns: the right child of self, None if there is no right child
    """

    def getRight(self):
        if self.right is None:
            self.right = AVLNode(None, True)
        return self.right

    """returns the parent 
    @rtype: AVLNode
    @returns: the parent of self, None if there is no parent
    """

    def getParent(self):
        return self.parent

    """return the value
    @rtype: str
    @returns: the value of self, None if the node is virtual
    """

    def getValue(self):
        return self.value

    """returns the height
    @rtype: int
    @returns: the height of self, -1 if the node is virtual
    """

    def getHeight(self):
        return self.height

    """sets left child
    @type node: AVLNode
    @param node: a node
    """

    def setLeft(self, node, setParent=True):
        if node is None:
            self.left = AVLNode(None, True)
            return
        self.left = node
        if setParent:
            node.setParent(self)

    """sets right child
    @type node: AVLNode
    @param node: a node
    """

    def setRight(self, node, setParent=True):
        if node is None:
            self.right = AVLNode(None, True)
            return
        self.right = node
        if setParent:
            node.setParent(self)

    """sets parent
    @type node: AVLNode
    @param node: a node
    """

    def setParent(self, node):
        self.parent = node

    """sets value
    @type value: str
    @param value: data
    """

    def setValue(self, value):
        self.value = value

    """sets the balance factor of the node
    @type h: int
    @param h: the height
    """

    def setHeight(self, h):
        self.height = h

    """returns whether self is not a virtual node 
    @rtype: bool
    @returns: False if self is a virtual node, True otherwise.
    """

    def isRealNode(self):
        return not self.is_virtual

    """returns the size of the current node 
    @rtype: int
    @returns: self.size
    """

    def getSize(self):
        return self.size

    """sets the size of the current node 
    @rtype: Void
    """

    def setSize(self, val):
        self.size = val

    #Part - Hieght and Size fixes
    """calculating the height of self and setting if needed
	@rtype: int
	@returns: 1 if height should be changed else 0
	"""

    def fixingHeight(self):
        newHeight = max(self.getLeft().getHeight(), self.getRight().getHeight()) + 1
        if self.getHeight() == newHeight:
            return 0
        self.setHeight(newHeight)
        return 1

    """calculating the size of self and setting if needed
	@rtype: int
	@returns:  1 if size should be changed else 0
	"""

    def fixingSize(self):
        newSize = self.getLeft().getSize() + self.getRight().getSize() + 1
        if newSize == self.getSize():
            return 0
        self.setSize(newSize)
        return 1

    """does right rotation on self
    @rtype: int
    @returns:  1 
    """
    # region - rotations
    def rightRotate(self):
        Y = self
        X = Y.getLeft()
        X_R = X.getRight()
        Y.setLeft(X_R, False)
        X_R.setParent(Y)
        X.setRight(Y, False)
        X.setParent(Y.parent)
        if Y.parent is not None:
            if Y.parent.getLeft() == Y:
                Y.parent.setLeft(X, False)
            else:
                Y.parent.setRight(X, False)
        Y.setParent(X)
        Y.fixingHeight()
        Y.fixingSize()
        X.fixingHeight()
        X.fixingSize()
        return 1

    """does left rotation from self
    @rtype: int
    @returns:  1 
    """

    def leftRotate(self):
        Y = self
        X = Y.getRight()
        X_L = X.getLeft()
        Y.setRight(X_L, False)
        X_L.setParent(Y)
        X.setLeft(Y, False)
        X.setParent(Y.parent)
        if Y.parent is not None:
            if Y.parent.getLeft() == Y:
                Y.parent.setLeft(X, False)
            else:
                Y.parent.setRight(X, False)
        Y.setParent(X)
        Y.fixingHeight()
        Y.fixingSize()
        X.fixingSize()
        X.fixingHeight()
        return 1

    """does right then left rotation from self
    @rtype: int
    @returns:  1 
    """

    def rightThenLeftRotate(self):
        self.getRight().rightRotate()
        self.leftRotate()
        return 2

    """does left then right rotation from self
    @rtype: int
    @returns:  2 
    """

    def leftThenRightRotate(self):
        self.getLeft().leftRotate()
        self.rightRotate()
        return 2

    """fixing the balance factors of the AVL tree using rotations from a given node all way up in recursive manner 
	@rtype: int
	@returns:  num of rotations that used fixing tree
	TimeComplexity: O(log(n))
	"""

    def fixingTree(self):
        def fixingTreeRec(node):
            cnt = 0
            if node is None:
                return 0
            cntHeightFix = node.fixingHeight()
            node.fixingSize()
            if node.calcBalanceFactor() > 1:
                if node.getLeft().calcBalanceFactor() == -1:
                    cnt += node.leftThenRightRotate()
                else:
                    cnt += node.rightRotate()
            elif node.calcBalanceFactor() < -1:
                if node.getRight().calcBalanceFactor() == 1:
                    cnt += node.rightThenLeftRotate()
                else:
                    cnt += node.leftRotate()
            else:
                cnt += cntHeightFix
            return cnt + fixingTreeRec(node.getParent())

        return fixingTreeRec(self)

    #Predecessor and Successor Part

    """finding node successor
	@rtype: AVLNode
	@returns: node successor
	TimeComplexity: O(log(n)) 
	"""

    def successor(self):
        def successorRec(node):
            if node.getRight().isRealNode():
                node = node.getRight()
                while node.getLeft().isRealNode():
                    node = node.getLeft()
                return node
            else:
                while node.rightChildCheck(node.getParent()):
                    node = node.parent
                return node.parent

        return successorRec(self)

    """finding node predecessor
	@rtype: AVLNode
	@returns: node predecessor
	TimeComplexity: O(log(n)) 
	"""

    def predecessor(self):
        def predecessorRec(node):
            if node.getLeft().isRealNode():
                node = node.getLeft()
                while node.getRight().isRealNode():
                    node = node.getRight()
                return node
            else:
                while node.leftChildCheck(node.getParent()):
                    node = node.getParent()
                return node.getParent()

        return predecessorRec(self)

"""
A class implementing the ADT list, using an AVL tree.
"""


class AVLTreeList(object):
    """
    Constructor, you are allowed to add more fields.
    """

    def __init__(self):
        self.size = 0
        self.root = None
        self.firstNode = None
        self.lastNode = None

    """returns the size of the AVLTree
    @rtype: int
    @returns: the size of self
    """

    def getSize(self):
        return self.size

    """sets the size of self
    @type s: int
    @param s: the size
    """

    def setSize(self):
        self.size = self.root.getSize()

    """returns the root of the tree representing the list
     @rtype: AVLNode
     @returns: the root, None if the list is empty
     """

    def getRoot(self):
        return self.root

    """returns whether the list is empty
    @rtype: bool
    @returns: True if the list is empty, False otherwise
    """

    def empty(self):
        return self.getRoot() is None or self.getRoot().getSize() == 0

    """returns the k'th item in the list
	@rtype: any
	@returns: k'th item in the list
	TimeComplexity: O(log(n)) 
	"""

    def select(self, k):
        node = self.getRoot()
        while node is not None and node.isRealNode():
            if not node.getLeft().isRealNode:
                m = 1
            else:
                m = node.getLeft().getSize() + 1
            if k == m:
                return node
            elif k < m:
                node = node.getLeft()
            else:
                node = node.getRight()
                k = k - m
        return node

    """retrieving the node of the i'th item in the list
	@type i: int
	@pre: 0 <= i < self.length()
	@param i: index in the list
	@rtype: AVLNode
	@returns: the the node of the i'th item in the list
	Time Complexity: O(log(n))
	"""

    def retrieveNode(self, i):
        return self.select(i + 1)

    """retrieving the value of the i'th item in the list
	@type i: int
	@pre: 0 <= i < self.length()
	@param i: index in the list
	@rtype: str
	@returns: the the value of the i'th item in the list
	Time Complexity: O(log(n))
	"""

    def retrieve(self, i):
        node = self.select(i + 1)
        return node.getValue() if node is not None else None

    """inserts val at position i in the list
    @type i: int
    @pre: 0 <= i <= self.length()
    @param i: The intended index in the list to which we insert val
    @type val: str
    @param val: the value we inserts
    @rtype: list
    @returns: the number of rebalancing operation due to AVL rebalancing
    Time Complexity: O(log(n))
    """

    def insert(self, i, val):
        added_node = AVLNode(val, False)
        added_node.addingVirtualNodes()
        if self.empty():
            self.root = added_node
            self.firstNode = self.getRoot()
            self.lastNode = self.getRoot()
            self.setSize()
            return 0
        else:
            if i == self.length():
                self.retrieveNode(i - 1).setRight(added_node)
                self.lastNode = added_node
            elif i < self.length():
                next_node = self.retrieveNode(i)
                if next_node.getLeft().isRealNode() is False:
                    next_node.setLeft(added_node)
                else:
                    pred = next_node.predecessor()
                    pred.setRight(added_node)
            if i == 0:
                self.firstNode = added_node
        if added_node.getParent() is not None:
            num_fixes = added_node.getParent().fixingTree()
        else:
            num_fixes = 0
        self.root = self.root.findingRoot()
        self.setSize()
        return num_fixes

    """deletes the i'th item in the list
    @type i: int
    @pre: 0 <= i < self.length()
    @param i: The intended index in the list to be deleted
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    Time Complexity: O(log(n))
    """

    def delete(self, i):
        num_fixes = 0
        if self.empty():
            return -1
        if i == 0 and self.getRoot().size == 1:
            self.root = None
            self.firstNode = None
            self.lastNode = None
            return 0
        deleted_node = self.retrieveNode(i)
        if deleted_node.getLeft().isRealNode() and deleted_node.getRight().isRealNode():
            successor = deleted_node.successor()
            deleted_node.setValue(successor.getValue())
            deleted_node = successor
        parent_node = deleted_node.parent
        replaced_node = deleted_node.getRight()
        if replaced_node is None or not replaced_node.isRealNode():
            replaced_node = deleted_node.getLeft() if deleted_node.getLeft().isRealNode() else None
        if parent_node is None:
            self.root = replaced_node
            deleted_node.setParent(None)
            deleted_node.setLeft(None)
            deleted_node.setRight(None)
            self.getRoot().setParent(None)
            num_fixes += self.root.fixingHeight()
            self.root.fixingSize()
        else:
            if deleted_node.rightChildCheck(parent_node):
                parent_node.setRight(replaced_node)
            elif deleted_node.leftChildCheck(parent_node):
                parent_node.setLeft(replaced_node)
            deleted_node.setParent(None)
            deleted_node.setLeft(None)
            deleted_node.setRight(None)
            num_fixes += parent_node.fixingTree()
            self.root = self.root.findingRoot()
        if i == self.length():
            self.lastNode = self.retrieveNode(self.length() - 1)
        if i == 0:
            self.firstNode = self.retrieveNode(0)
        self.setSize()
        return num_fixes

    """returns the value of the first item in the list
    @rtype: str
    @returns: the value of the first item, None if the list is empty
    """

    def first(self):
        return self.firstNode.getValue() if self.firstNode is not None else None

    """returns the value of the last item in the list
    @rtype: str
    @returns: the value of the last item, None if the list is empty
    """

    def last(self):
        return self.lastNode.getValue() if self.lastNode is not None else None

    """returns an array representing list 
    @rtype: list
    @returns: a list of strings representing the data structure
    Time Complexity: O(n)
    """

    def listToArray(self):
        def listToArrayRec(currNode, arr):
            if currNode is None or not currNode.isRealNode():
                return
            listToArrayRec(currNode.getLeft(), arr)
            arr.append(currNode.value)
            listToArrayRec(currNode.getRight(), arr)
        res = []
        listToArrayRec(self.getRoot(), res)
        return res

    """returns the size of the list 
    @rtype: int
    @returns: the size of the list
    """

    def length(self):
        return self.root.getSize() if not self.empty() else 0

    """sort the array in place by dividing it to two sub arrays
    @rtype: Void
    Time Complexity: O(nlog(n))
    """

    def mergeSort(self, arr):
        if len(arr) > 1:
            mid = len(arr) // 2
            sub_array1 = arr[:mid]
            sub_array2 = arr[mid:]
            # Sort the two halves
            self.mergeSort(sub_array1)
            self.mergeSort(sub_array2)
            i = j = k = 0
            while i < len(sub_array1) and j < len(sub_array2):
                if sub_array1[i] < sub_array2[j]:
                    arr[k] = sub_array1[i]
                    i += 1
                else:
                    arr[k] = sub_array2[j]
                    j += 1
                k += 1
            while i < len(sub_array1):
                arr[k] = sub_array1[i]
                i += 1
                k += 1

            while j < len(sub_array2):
                arr[k] = sub_array2[j]
                j += 1
                k += 1

    """sort the info values of the list using merge sort
    @rtype: list
    @returns: an AVLTreeList where the values are sorted by the info of the original list.
    Time Complexity: O(nlog(n))
    """

    def sort(self):
        arr = self.listToArray();
        self.mergeSort(arr);
        sorted_tree = AVLTreeList();
        for i in range(len(arr)):
            sorted_tree.insert(i, arr[i])
        return sorted_tree

    """permute the info values of the list 
    @rtype: list
    @returns: an AVLTreeList where the values are permuted randomly by the info of the original list. ##Use Randomness
    Time Complexity: O(nlog(n))
    """
    def permutation(self):
        permuted_tree = AVLTreeList();
        arr = self.listToArray();
        len_arr = len(arr);
        for i in range(len_arr):
            index = random.randrange(0, len_arr - i);
            permuted_tree.insert(i, arr[index]);
            del arr[index];
        return permuted_tree

    """join lst and self using x as a connecting node
	@type bridge: AVLNode
	@type lst: AVLTreeList
	@param x: a node to connect between self and lst
	@param lst: a list to concat to self using join
	@rtype: Void
	Time Complexity: O(log(delta(h))) - delta(h) represents the heights' difference between self tree and lst tree
	"""

    def join(self, x, lst):
        if lst.empty() or lst is None:  # in this case x added as last node of self
            len = self.length()
            self.insert(len, x.getValue())
            self.lastNode = self.retrieveNode(len - 1)
            return
        if self.empty():  # in this case x added as first node of self
            self.firstNode = x

        if self.getRoot() is None or lst.getRoot().getHeight() >= self.getRoot().getHeight():  # lst height is equal or higher than self height
            node_to_connect = lst.getRoot()
            left_height = self.getRoot().getHeight() if not self.empty() else 0
            while node_to_connect.getLeft().isRealNode and left_height < node_to_connect.getHeight():
                node_to_connect = node_to_connect.getLeft()
            x.setLeft(self.getRoot())
            x.addingVirtualNodes()
            if node_to_connect.getParent() is not None:
                node_to_connect.getParent().setLeft(x)
                self.root = lst.getRoot()
            else:
                self.root = x
            x.setRight(node_to_connect)

        else:  # self height is higher than lst height
            node_to_connect = self.getRoot()
            while node_to_connect.getRight().isRealNode() and lst.getRoot().getHeight() < node_to_connect.getHeight():
                node_to_connect = node_to_connect.getRight()
            x.setRight(lst.getRoot())
            if node_to_connect.getParent() is not None:
                node_to_connect.getParent().setRight(x)
            else:
                self.root = x
            x.setLeft(node_to_connect)
        x.fixingTree()
        self.root = self.root.findingRoot()
        self.firstNode = self.firstNode
        self.lastNode = lst.lastNode
        self.setSize()

    """concatenates lst to self
    @type lst: AVLTreeList
    @param lst: a list to be concatenated after self
    @rtype: int
    @returns: the absolute value of the difference between the height of the AVL trees joined
    Time Complexity: O(log n)
    """

    def concat(self, lst):
        if self.empty() and lst.empty():
            return 0
        if lst.empty():
            return self.getRoot().getHeight() + 1  # the height of empty list is -1
        if self.empty():
            self.root = lst.getRoot()
            self.firstNode = lst.firstNode
            self.lastNode = lst.lastNode
            return self.getRoot().getHeight() + 1
        difference = abs(lst.getRoot().getHeight() - self.getRoot().getHeight())
        x = self.retrieveNode(self.length() - 1)
        self.delete(self.length() - 1)
        self.join(x, lst)
        return difference


    """searches for a *value* in the list
    @type val: str
    @param val: a value to be searched
    @rtype: int
    @returns: the first index that contains val, -1 if not found.
    Time Complexity: O(n)
    """

    def search(self, val):
        def searchRec(j, val, searched_node):
            if searched_node is None or not searched_node.isRealNode():
                return j, -1
            (curr_ind, look_ind) = searchRec(j, val, searched_node.getLeft())
            curr_ind += 1
            if look_ind > -1:
                return curr_ind, look_ind
            if searched_node.getValue() == val:
                return curr_ind, curr_ind
            return searchRec(curr_ind, val, searched_node.getRight())

        (k, ind) = searchRec(-1, val, self.getRoot())
        return ind



