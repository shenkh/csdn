# IOU

图像处理的交并比（IoU） https://www.lao-wang.com/?p=114

```cpp
struct rect{
    int x1; //左上
    int y1;
    int x2; //右下
    int y2;
    int area(){
        return (y2-y1)*(x2-x1);
    }
    rect(int left, int top, int right, int down) :
        x1(left), y1(top), x2(right), y2(down){}
};

float iou(rect c, rect gt){
    int ix1 = max(c.x1, gt.x1);
    int iy1 = max(c.y1, gt.y1);
    int ix2 = min(c.x2, gt.x2);
    int iy2 = min(c.y2, gt.y2);

    int w = max(ix2-ix1, 0);
    int h = max(iy2-iy1, 0);

    int area = w*h;
    float iou = area / (float)(c.area + gt.area - area)
    return iou;
}
```