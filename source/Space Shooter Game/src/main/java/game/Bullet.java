// Bullet.java
package game;

import java.awt.*;

public class Bullet extends GameObject {
    private static final int SPEED = 10;
    private boolean isPlayerBullet;
    private int damage;
    private WeaponType weaponType;
    private double angle = 0;

    public enum WeaponType {
        NORMAL("💫", 10, 1),
        DOUBLE("⭐", 15, 2),
        TRIPLE("🌟", 25, 3),
        LASER("☄️", 40, 1),
        SPREAD("✨", 20, 3);

        private final String emoji;
        private final int damage;
        private final int projectileCount;

        WeaponType(String emoji, int damage, int projectileCount) {
            this.emoji = emoji;
            this.damage = damage;
            this.projectileCount = projectileCount;
        }
    }

    public Bullet(double x, double y, boolean isPlayerBullet, WeaponType weaponType) {
        super(x, y, 5, 10);
        this.isPlayerBullet = isPlayerBullet;
        this.weaponType = weaponType;
        this.damage = weaponType.damage;
    }

    public Bullet(double x, double y, boolean isPlayerBullet, WeaponType weaponType, double angle) {
        this(x, y, isPlayerBullet, weaponType);
        this.angle = angle;
    }

    @Override
    public void update() {
        if (weaponType == WeaponType.LASER) {
            // 激光直线向上
            y -= SPEED * 1.5;
        } else {
            // 其他子弹可以有角度
            x += Math.sin(angle) * SPEED;
            y -= Math.cos(angle) * SPEED * (isPlayerBullet ? 1 : -1);
        }

        if (y < 0 || y > 600 || x < 0 || x > 800) {
            active = false;
        }
    }

    @Override
    public void render(Graphics2D g) {
        g.setFont(new Font("Arial", Font.PLAIN, 20));
        g.drawString(isPlayerBullet ? weaponType.emoji : "💥", (int)x, (int)y + height);

        if (weaponType == WeaponType.LASER) {
            // 绘制激光效果
            g.setColor(new Color(255, 0, 0, 100));
            g.fillRect((int)x - 2, (int)y - 20, 4, 30);
        }
    }

    public boolean isPlayerBullet() { return isPlayerBullet; }
    public int getDamage() { return damage; }
    public WeaponType getWeaponType() { return weaponType; }
}