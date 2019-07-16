"# taxonomy10k" 

Formatting Rules
1.	Each node is marked by ‘-‘. The name of the node as well as all attributes are listed under one ‘-‘
2.	Leaf nodes are written as “- name: NodeName” No matter what file they are in.
3.	Parent nodes that have their children in a sub file are written as follows where the name of the node matches the sub file name.
- NodeName: []
  subfile: NodeName.yaml
4.	Parent nodes that have their children in the same file will be written as follows
- NodeName:
   - child name
   - child name
5.	If a parent node is also a leaf node it will be relisted under itself with leaf node structure. For instance if we plan to have the object “tree” as well as tree types such as “oak tree” it will look like
- tree:
    - name: tree
      noffset: n13104059
      url: http://image-net.org/synset?wnid=n13104059
      bestQuery: [tree, small tree, big tree, trees in fall]
   - name: oak tree
