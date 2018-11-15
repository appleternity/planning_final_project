import cv2
import numpy as np

p_color = (255, 255, 0)
d_color = (100, 100, 100)
c_color = (255, 255, 255)

def same_color(p1, p2):
    if len(p1) != len(p2): return False
    for pp1, pp2 in zip(p1, p2):
        if pp1 != pp2: return False
    return True

class Display:
    def __init__(self, graph, mapping, map_type, k, w, fix_r, fix_c, unit=40):
        self.graph = graph # {"0":[1, 9], ...}
        self.mapping = mapping # {"0":("x", "y"), ...}
        
        # TODO: Notice the order
        self.graph = {int(k):v for k, v in self.graph.items()}
        self.mapping = {int(k):(int(v[0]), int(v[1])) for k, v in self.mapping.items()}

        scale = 2
        self.height = 230*scale
        self.width  = 370*scale
        self.unit   = unit*scale

        create_graph = {
            "tree":self.create_tree,
            "ladder":self.create_ladder,
            "window":self.create_window,
        }
        if map_type not in create_graph:
            raise "May Type Not Found"

        self.img = create_graph[map_type](k, w)
        self.mask = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def create_tree(self, k, w, fix_r=20, fix_c=20):
        img = np.zeros((self.height, self.width, 3), np.uint8)
        cv2.line(img, (0, int(self.unit/2)), (self.width, int(self.unit/2)), d_color, self.unit)
        for i in range(0, k):
            x = int((self.width-k*self.unit*w)/(k+1) * (i+1) + self.unit*i*w + w*self.unit/2)
            cv2.line(img, (x, 0), (x, self.height), d_color, w*self.unit)
        return img

    def create_ladder(self, k, w, fix_r=20, fix_c=20):
        img = np.zeros((self.height, self.width, 3), np.uint8)
        cv2.line(img, (0, int(self.unit/2)), (self.width, int(self.unit/2)), d_color, self.unit)
        cv2.line(img, (0, int(self.height-self.unit/2)), (self.width, int(self.height-self.unit/2)), d_color, self.unit)
        for i in range(0, k):
            x = int((self.width-k*self.unit*w) / (k-1) * i + self.unit*i*w + w*self.unit/2)
            cv2.line(img, (x, 0), (x, self.height), d_color, w*self.unit)
        return img

    def create_window(self, k, w, fix_r=20, fix_c=20):
        img = np.zeros((self.height, self.width, 3), np.uint8)
        cv2.line(img, (0, int(self.unit/2)), (self.width, int(self.unit/2)), d_color, self.unit)
        cv2.line(img, (0, int(self.height-self.unit/2)), (self.width, int(self.height-self.unit/2)), d_color, self.unit)
        cv2.line(img, (0, int(self.height/2)), (self.width, int(self.height/2)), d_color, self.unit)
        for i in range(0, k):
            x = int((self.width-k*self.unit*w) / (k-1) * i + self.unit*i*w + w*self.unit/2)
            cv2.line(img, (x, 0), (x, self.height), d_color, w*self.unit)
        return img

    def draw(self, state, t=30):
        img = np.copy(self.img)
        #pursuers = state.pursuers
       
        #dirty = state.dirty
        dirty = state.g
        mask = np.zeros((self.height+2, self.width+2), np.uint8)
        for d, n in zip(dirty, self.graph):
            #if d is 1: continue
            if d is -1: continue
            x, y = self.mapping[n]
            if same_color(img[x, y], p_color): continue
            #cv2.floodFill(img, mask, (y, x), c_color)
            cv2.circle(img, (y, x), self.unit, c_color, -1)
        
        if state.keep:
            pursuers = state.pursuers
        else:
            pursuers = [i for i, g in enumerate(state.g) if g >= 1]
        for p_idx, p in enumerate(pursuers):
            x, y = self.mapping[p]
            if state.keep:
                color = (255 - p_idx * 12, 255 - p_idx * 35, p_idx * 40)
                cv2.circle(img, (y, x), self.unit, color, -1)
            else:
                cv2.circle(img, (y, x), self.unit, p_color, -1)
        
        img[self.mask==0] = (0, 0, 0)
        cv2.imshow("test", img)
        cv2.waitKey(t)

def display_test():
    import json
    import os.path
    from reinforcement_learning import State
    """
    filename = "_ladder_k2_w1"
    
    with open(os.path.join("mapping", filename+"_mapping_dictionary.json"), 'r') as infile:
        mapping = json.load(infile)

    with open(os.path.join("state", filename+"_state.json"), 'r') as infile:
        graph = json.load(infile)

    display = Display(graph, mapping, "ladder", k=2, w=1, fix_r=45, fix_c=50)
    #state = State((0, 6), (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
    state = State((13, 14), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    display.draw(state)
    """
    filename = "_tree_k1_w2"
    
    with open(os.path.join("mapping", filename+"_mapping_dictionary.json"), 'r') as infile:
        mapping = json.load(infile)

    with open(os.path.join("state", filename+"_state.json"), 'r') as infile:
        graph = json.load(infile)

    display = Display(graph, mapping, "tree", k=1, w=2, fix_r=50, fix_c=70)
    #state = State((0, 6), (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))
    state = State((9, 10), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
    display.draw(state)

def test():
    img = np.zeros((200, 300, 3), np.uint8)
    mask = np.zeros((202, 302), np.uint8)
    cv2.line(img, (150, 0), (150, 200), (0, 255, 255), 10)
    
    cv2.floodFill(img, mask, (100, 100), (255, 255, 0))
    cv2.floodFill(img, mask, (155, 30), (0, 255, 0))
    
    cv2.imshow("test", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    #test()
    display_test()
