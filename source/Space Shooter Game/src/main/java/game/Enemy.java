package game;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Enemy extends GameObject {
    private double speed;
    private int health;
    private EnemyType type;
    private long lastShootTime;
    private static final Random random = new Random();
    private double movementAngle = 0;
    private boolean movingHorizontally = false;
    private List<Bullet> bullets = new ArrayList<>();

    public enum EnemyType {
        BASIC("👾", 20, 2.0, false),
        TANK("🛸", 50, 1.5, false),
        SHOOTER("👽", 30, 2.5, true),
        BOSS("👿", 1000, 1.0, true);

        private final String emoji;
        private final int health;
        private final double speed;
        private final boolean canShoot;

        EnemyType(String emoji, int health, double speed, boolean canShoot) {
            this.emoji = emoji;
            this.health = health;
            this.speed = speed;
            this.canShoot = canShoot;
        }
    }

    public Enemy(double x, double y, EnemyType type) {
        super(x, y, 30, 30);
        this.type = type;
        this.health = type.health;
        this.speed = type.speed;
        this.movingHorizontally = random.nextBoolean();
        if (type == EnemyType.BOSS) {
            this.width = 60;
            this.height = 60;
        }
    }

    @Override
    public void update() {
        // 默认的update方法，在Game类中调用带参数的update
        update(bullets);
    }

    public void update(List<Bullet> bullets) {
        this.bullets = bullets; // 保存子弹列表引用
        if (type == EnemyType.BOSS) {
            updateBoss();
        } else {
            updateNormal();
        }

        if (type.canShoot && System.currentTimeMillis() - lastShootTime > 2000) {
            shoot(bullets);
            lastShootTime = System.currentTimeMillis();
        }
    }

    private void updateNormal() {
        if (movingHorizontally) {
            x += Math.sin(movementAngle) * speed;
            if (x <= 0 || x >= 800 - width) {
                movementAngle = Math.PI - movementAngle;
            }
        }
        y += speed;

        if (y > 600) {
            active = false;
        }
    }

    private void updateBoss() {
        x += Math.sin(movementAngle) * speed;
        if (x <= 0 || x >= 800 - width) {
            movementAngle = Math.PI - movementAngle;
        }
        if (y < 100) {
            y += speed;
        }
    }

    private void shoot(List<Bullet> bullets) {
        if (type == EnemyType.BOSS) {
            for (int i = 0; i < 5; i++) {
                double angle = (Math.PI / 4) * (i - 2);
                bullets.add(new Bullet(x + width/2, y + height,
                        false, Bullet.WeaponType.SPREAD, angle));
            }
        } else {
            bullets.add(new Bullet(x + width/2, y + height,
                    false, Bullet.WeaponType.NORMAL));
        }
    }

    @Override
    public void render(Graphics2D g) {
        // 绘制敌人
        g.setFont(new Font("Arial", Font.PLAIN, type == EnemyType.BOSS ? 50 : 30));
        g.drawString(type.emoji, (int)x, (int)y + height);

        // Boss血条
        if (type == EnemyType.BOSS) {
            drawBossHealthBar(g);
        }
    }

    private void drawBossHealthBar(Graphics2D g) {
        g.setColor(Color.RED);
        g.fillRect((int)x, (int)y - 10, (int)(width * health / type.health), 5);
        g.setColor(Color.WHITE);
        g.drawRect((int)x, (int)y - 10, width, 5);
    }

    public void damage(int amount) {
        health -= amount;
        if (health <= 0) {
            active = false;
        }
    }

    public EnemyType getType() { return type; }
    public int getHealth() { return health; }
}