package game;

import java.awt.*;

public abstract class GameObject {
    protected double x, y;
    protected double dx, dy;
    protected int width, height;
    protected boolean active = true;

    public GameObject(double x, double y, int width, int height) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    public abstract void update();
    public abstract void render(Graphics2D g);

    public boolean intersects(GameObject other) {
        return x < other.x + other.width &&
                x + width > other.x &&
                y < other.y + other.height &&
                y + height > other.y;
    }

    public double getX() { return x; }
    public double getY() { return y; }
    public boolean isActive() { return active; }
}
