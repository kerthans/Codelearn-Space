package game;

import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.geom.Rectangle2D;

public class Player extends GameObject {
    private static final int BASE_SPEED = 5;
    private int health = 100;
    private int maxHealth = 100;
    private int score = 0;
    private int weaponLevel = 1;
    private int shield = 0;
    private boolean[] keys;
    private boolean laserUnlocked = false;
    private long lastDashTime = 0;
    private static final long DASH_COOLDOWN = 1000;
    private boolean isDashing = false;
    private String currentSprite = "🚀";
    private Color shieldColor = new Color(64, 224, 208, 100);

    public Player(double x, double y) {
        super(x, y, 40, 40);
        keys = new boolean[256];
    }

    @Override
    public void update() {
        updateMovement();
        updateEffects();
    }

    private void updateMovement() {
        double speed = BASE_SPEED + Math.min(weaponLevel - 1, 3);
        if (isDashing) speed *= 2;

        if (keys[KeyEvent.VK_W] || keys[KeyEvent.VK_UP]) y -= speed;
        if (keys[KeyEvent.VK_S] || keys[KeyEvent.VK_DOWN]) y += speed;
        if (keys[KeyEvent.VK_A] || keys[KeyEvent.VK_LEFT]) x -= speed;
        if (keys[KeyEvent.VK_D] || keys[KeyEvent.VK_RIGHT]) x += speed;

        // Dash ability
        if (keys[KeyEvent.VK_SHIFT] && System.currentTimeMillis() - lastDashTime > DASH_COOLDOWN) {
            performDash();
        }

        x = Math.max(0, Math.min(x, 800 - width));
        y = Math.max(0, Math.min(y, 600 - height));
    }

    private void updateEffects() {
        if (shield > 0) shield--;
        if (isDashing && System.currentTimeMillis() - lastDashTime > 200) {
            isDashing = false;
        }
        // 根据状态更新外观
        updateSprite();
    }

    private void performDash() {
        isDashing = true;
        lastDashTime = System.currentTimeMillis();
        if (shield <= 0) shield = 50; // 短暂无敌
    }

    private void updateSprite() {
        if (isDashing) {
            currentSprite = "⚡";
        } else if (health < 30) {
            currentSprite = "🔥";
        } else if (weaponLevel >= 4) {
            currentSprite = "🛸";
        } else {
            currentSprite = "🚀";
        }
    }

    @Override
    public void render(Graphics2D g) {
        // 绘制护盾效果
        if (shield > 0) {
            g.setColor(shieldColor);
            g.fillOval((int)x - 5, (int)y - 5, width + 10, height + 10);
        }

        // 绘制玩家
        g.setFont(new Font("Arial", Font.PLAIN, 40));
        g.setColor(Color.WHITE);
        g.drawString(currentSprite, (int)x, (int)y + height);

        // 绘制血条
        drawHealthBar(g);
    }

    private void drawHealthBar(Graphics2D g) {
        int barWidth = 40;
        int barHeight = 4;
        int x = (int)this.x;
        int y = (int)this.y - 10;

        // 背景
        g.setColor(Color.RED);
        g.fillRect(x, y, barWidth, barHeight);

        // 血量
        g.setColor(Color.GREEN);
        int healthWidth = (int)((health / (float)maxHealth) * barWidth);
        g.fillRect(x, y, healthWidth, barHeight);
    }

    public void heal(int amount) {
        health = Math.min(maxHealth, health + amount);
    }

    public void damage(int amount) {
        if (shield > 0) {
            shield = Math.max(0, shield - amount);
            return;
        }
        health = Math.max(0, health - amount);
        if (health <= 0) {
            active = false;
        }
    }

    public void addShield(int amount) {
        shield = Math.min(100, shield + amount);
    }

    public void unlockLaser() {
        laserUnlocked = true;
    }

    public void setHealth(int health) {
        this.health = Math.min(maxHealth, health);
    }

    public void setKey(int keyCode, boolean pressed) {
        if (keyCode >= 0 && keyCode < keys.length) {
            keys[keyCode] = pressed;
        }
    }

    public void upgradeWeapon() {
        weaponLevel = Math.min(5, weaponLevel + 1);
    }

    // Getters
    public int getHealth() { return health; }
    public int getScore() { return score; }
    public int getWeaponLevel() { return weaponLevel; }
    public int getShield() { return shield; }
    public boolean isLaserUnlocked() { return laserUnlocked; }
    public Rectangle2D.Double getBounds() {
        return new Rectangle2D.Double(x, y, width, height);
    }
}