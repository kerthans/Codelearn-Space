// Game.java
package game;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Game extends JPanel implements ActionListener, KeyListener {
    private static final int WIDTH = 800;
    private static final int HEIGHT = 600;
    private static final Color BACKGROUND_COLOR = new Color(0, 0, 20);

    private Player player;
    private List<Enemy> enemies;
    private List<Bullet> bullets;
    private List<PowerUp> powerUps;
    private Level currentLevel;
    private int score = 0;
    private int highScore = 0;
    private boolean gameOver = false;
    private boolean gamePaused = false;
    private boolean gameStarted = false;
    private boolean levelCompleted = false;
    private int frameCount = 0;
    private final Random random;
    private long lastBossAttackTime = 0;
    private Rectangle dangerZone;
    private boolean chargingLaser = false;
    private int chargeTime = 0;

    // UI elements
    private JButton startButton;
    private JButton restartButton;
    private JButton nextLevelButton;

    public Game() {
        setPreferredSize(new Dimension(WIDTH, HEIGHT));
        setBackground(BACKGROUND_COLOR);
        setLayout(null);
        setFocusable(true);
        addKeyListener(this);

        initializeUI();
        random = new Random();

        Timer timer = new Timer(16, this);
        timer.start();
    }

    private void initializeUI() {
        startButton = new JButton("开始游戏 🎮");
        startButton.setBounds(WIDTH / 2 - 100, HEIGHT / 2 - 25, 200, 50);
        startButton.addActionListener(e -> startGame());
        add(startButton);

        restartButton = new JButton("重新开始 🔄");
        restartButton.setBounds(WIDTH / 2 - 100, HEIGHT / 2 + 50, 200, 50);
        restartButton.setVisible(false);
        restartButton.addActionListener(e -> restartGame());
        add(restartButton);

        nextLevelButton = new JButton("下一关 ➡️");
        nextLevelButton.setBounds(WIDTH / 2 - 100, HEIGHT / 2 + 50, 200, 50);
        nextLevelButton.setVisible(false);
        nextLevelButton.addActionListener(e -> startNextLevel());
        add(nextLevelButton);
    }

    private void startGame() {
        gameStarted = true;
        gameOver = false;
        levelCompleted = false;
        startButton.setVisible(false);
        score = 0;

        player = new Player(WIDTH / 2, HEIGHT - 100);
        enemies = new ArrayList<>();
        bullets = new ArrayList<>();
        powerUps = new ArrayList<>();
        currentLevel = new Level(1);

        requestFocusInWindow();
    }

    private void startNextLevel() {
        levelCompleted = false;
        nextLevelButton.setVisible(false);
        currentLevel = new Level(currentLevel.getLevelNumber() + 1);
        enemies.clear();
        bullets.clear();
        powerUps.clear();
        player.setHealth(100);
        frameCount = 0;

        if (currentLevel.isLaserUnlocked()) {
            player.unlockLaser();
        }

        requestFocusInWindow();
    }

    private void restartGame() {
        startGame();
        restartButton.setVisible(false);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        // 绘制星空背景
        drawStarryBackground(g2d);

        if (!gameStarted) {
            drawStartScreen(g2d);
            return;
        }

        if (gamePaused) {
            drawPauseScreen(g2d);
            return;
        }

        // 游戏对象
        if (player != null) player.render(g2d);
        enemies.forEach(enemy -> enemy.render(g2d));
        bullets.forEach(bullet -> bullet.render(g2d));
        powerUps.forEach(powerUp -> powerUp.render(g2d));

        // 危险区域
        if (dangerZone != null && currentLevel.getLevelNumber() == 5) {
            g2d.setColor(new Color(255, 0, 0, 100));
            g2d.fill(dangerZone);
        }

        // HUD
        drawHUD(g2d);

        if (levelCompleted) {
            drawLevelComplete(g2d);
        } else if (gameOver) {
            drawGameOver(g2d);
        }
    }

    private void drawStarryBackground(Graphics2D g) {
        g.setColor(Color.WHITE);
        for (int i = 0; i < 100; i++) {
            int x = (frameCount + random.nextInt(WIDTH)) % WIDTH;
            int y = random.nextInt(HEIGHT);
            g.fillRect(x, y, 1, 1);
        }
    }

    private void drawStartScreen(Graphics2D g) {
        g.setColor(Color.WHITE);
        g.setFont(new Font("Arial", Font.BOLD, 48));
        g.drawString("太空射击 🚀", WIDTH / 2 - 150, HEIGHT / 3);
        g.setFont(new Font("Arial", Font.PLAIN, 24));
        g.drawString("最高分: " + highScore, WIDTH / 2 - 80, HEIGHT / 3 + 50);
    }

    private void drawPauseScreen(Graphics2D g) {
        g.setColor(new Color(0, 0, 0, 150));
        g.fillRect(0, 0, WIDTH, HEIGHT);
        g.setColor(Color.WHITE);
        g.setFont(new Font("Arial", Font.BOLD, 48));
        g.drawString("暂停 ⏸️", WIDTH / 2 - 100, HEIGHT / 2);
    }

    private void drawHUD(Graphics2D g) {
        g.setColor(Color.WHITE);
        g.setFont(new Font("Arial", Font.BOLD, 20));
        g.drawString("分数: " + score, 10, 25);
        g.drawString("关卡: " + currentLevel.getLevelNumber() + " 🎯", 10, 50);
        g.drawString("目标分数: " + currentLevel.getTargetScore(), 10, 75);
        g.drawString("生命值: " + player.getHealth() + " ❤️", 10, 100);
        g.drawString("武器等级: " + player.getWeaponLevel() + " ⚔️", 10, 125);

        if (currentLevel.isLaserUnlocked()) {
            g.drawString("激光充能: " + (chargingLaser ? chargeTime/10 + "%" : "就绪 🔋"), 10, 150);
        }
    }

    private void drawLevelComplete(Graphics2D g) {
        g.setColor(new Color(0, 0, 0, 150));
        g.fillRect(0, 0, WIDTH, HEIGHT);
        g.setColor(Color.GREEN);
        g.setFont(new Font("Arial", Font.BOLD, 48));
        g.drawString("关卡完成! 🎉", WIDTH / 2 - 150, HEIGHT / 2 - 50);
        g.setFont(new Font("Arial", Font.BOLD, 24));
        if (currentLevel.getLevelNumber() == 2) {
            g.drawString("解锁激光武器! ⚡", WIDTH / 2 - 100, HEIGHT / 2);
        }
    }

    private void drawGameOver(Graphics2D g) {
        g.setColor(new Color(0, 0, 0, 150));
        g.fillRect(0, 0, WIDTH, HEIGHT);
        g.setColor(Color.RED);
        g.setFont(new Font("Arial", Font.BOLD, 48));
        g.drawString("游戏结束! 💔", WIDTH / 2 - 120, HEIGHT / 2 - 50);
        g.setFont(new Font("Arial", Font.BOLD, 24));
        g.drawString("最终分数: " + score, WIDTH / 2 - 80, HEIGHT / 2);
        if (score > highScore) {
            highScore = score;
            g.setColor(Color.YELLOW);
            g.drawString("新纪录! 🏆", WIDTH / 2 - 100, HEIGHT / 2 + 40);
        }
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        if (!gameStarted || gamePaused) return;

        if (!gameOver && !levelCompleted) {
            frameCount++;
            updateGame();
        }
        repaint();
    }

    private void updateGame() {
        player.update();
        updateEnemies();
        updateBullets();
        updatePowerUps();
        checkCollisions();
        updateBossAttacks();

        // 生成敌人和道具
        if (frameCount % currentLevel.getPowerUpFrequency() == 0) {
            spawnPowerUp();
        }

        if (frameCount % Math.max(20, 60 - currentLevel.getLevelNumber() * 10) == 0) {
            spawnEnemy();
        }

        // 检查关卡完成
        checkLevelCompletion();

        // 激光充能
        if (chargingLaser) {
            chargeTime = Math.min(chargeTime + 1, 1000);
        }
    }

    private void updateBossAttacks() {
        if (currentLevel.getLevelNumber() == 5 && !enemies.isEmpty()) {
            Enemy boss = enemies.stream()
                    .filter(e -> e.getType() == Enemy.EnemyType.BOSS)
                    .findFirst()
                    .orElse(null);

            if (boss != null && System.currentTimeMillis() - lastBossAttackTime > 3000) {
                // 创建危险区域
                int zoneType = random.nextInt(3);
                switch (zoneType) {
                    case 0: // 横向激光
                        dangerZone = new Rectangle(0, random.nextInt(HEIGHT - 100), WIDTH, 30);
                        break;
                    case 1: // 纵向激光
                        dangerZone = new Rectangle(random.nextInt(WIDTH - 30), 0, 30, HEIGHT);
                        break;
                    case 2: // 圆形冲击波
                        int centerX = random.nextInt(WIDTH);
                        int centerY = random.nextInt(HEIGHT/2);
                        dangerZone = new Rectangle(centerX - 50, centerY - 50, 100, 100);
                        break;
                }
                lastBossAttackTime = System.currentTimeMillis();
            }
        }
    }

    private void spawnEnemy() {
        if (enemies.size() >= currentLevel.getEnemyCount()) return;

        double x = random.nextDouble() * (WIDTH - 30);
        if (currentLevel.getLevelNumber() == 5 && enemies.isEmpty()) {
            // 生成Boss
            enemies.add(new Enemy(WIDTH/2 - 30, -50, Enemy.EnemyType.BOSS));
        } else {
            // 根据关卡生成不同类型的敌人
            Enemy.EnemyType type;
            if (currentLevel.getLevelNumber() >= 3 && random.nextDouble() < 0.2) {
                type = Enemy.EnemyType.TANK;
            } else if (currentLevel.getLevelNumber() >= 4 && random.nextDouble() < 0.3) {
                type = Enemy.EnemyType.SHOOTER;
            } else {
                type = Enemy.EnemyType.BASIC;
            }
            enemies.add(new Enemy(x, -30, type));
        }
    }

    private void spawnPowerUp() {
        double x = random.nextDouble() * (WIDTH - 30);
        PowerUp.PowerUpType[] types = PowerUp.PowerUpType.values();
        PowerUp.PowerUpType type = types[random.nextInt(types.length)];
        powerUps.add(new PowerUp(x, -30, type));
    }

    private void updateEnemies() {
        enemies.removeIf(enemy -> !enemy.isActive());
        enemies.forEach(enemy -> enemy.update(bullets));
    }

    private void updateBullets() {
        bullets.removeIf(bullet -> !bullet.isActive());
        bullets.forEach(Bullet::update);
    }

    private void updatePowerUps() {
        powerUps.removeIf(powerUp -> !powerUp.isActive());
        powerUps.forEach(PowerUp::update);
    }

    private void checkCollisions() {
        // 子弹与敌人碰撞
        for (Bullet bullet : bullets) {
            if (bullet.isPlayerBullet()) {
                for (Enemy enemy : enemies) {
                    if (bullet.intersects(enemy)) {
                        bullet.active = false;
                        enemy.damage(bullet.getDamage());
                        if (!enemy.isActive()) {
                            score += enemy.getType() == Enemy.EnemyType.BOSS ? 500 :
                                    enemy.getType() == Enemy.EnemyType.TANK ? 50 :
                                            enemy.getType() == Enemy.EnemyType.SHOOTER ? 30 : 10;
                        }
                    }
                }
            }
        }

        // 玩家与道具碰撞
        for (PowerUp powerUp : powerUps) {
            if (player.intersects(powerUp)) {
                applyPowerUp(powerUp);
                powerUp.active = false;
            }
        }

        // 玩家与敌人碰撞
        for (Enemy enemy : enemies) {
            if (player.intersects(enemy)) {
                player.damage(20);
                if (enemy.getType() != Enemy.EnemyType.BOSS) {
                    enemy.active = false;
                }
                if (!player.isActive()) {
                    gameOver = true;
                }
            }
        }

        // 敌人子弹与玩家碰撞
        for (Bullet bullet : bullets) {
            if (!bullet.isPlayerBullet() && bullet.intersects(player)) {
                player.damage(bullet.getDamage());
                bullet.active = false;
                if (!player.isActive()) {
                    gameOver = true;
                }
            }
        }

        // 危险区域伤害
        if (dangerZone != null && dangerZone.intersects(player.getBounds())) {
            player.damage(1);
            if (!player.isActive()) {
                gameOver = true;
            }
        }
    }

    private void checkLevelCompletion() {
        if (score >= currentLevel.getTargetScore() && !levelCompleted) {
            if (currentLevel.getLevelNumber() == 5) {
                // 检查Boss是否被击败
                boolean bossDefeated = enemies.stream()
                        .noneMatch(e -> e.getType() == Enemy.EnemyType.BOSS);
                if (bossDefeated) {
                    levelCompleted = true;
                    nextLevelButton.setVisible(true);
                    showVictoryEffects();
                }
            } else {
                levelCompleted = true;
                nextLevelButton.setVisible(true);
            }
        }
    }

    private void showVictoryEffects() {
        // TODO: 添加胜利特效，比如粒子效果或动画
    }

    private void applyPowerUp(PowerUp powerUp) {
        switch (powerUp.getType()) {
            case HEALTH:
                player.heal(30);
                break;
            case WEAPON:
                if (player.getWeaponLevel() < currentLevel.getMaxWeaponLevel()) {
                    player.upgradeWeapon();
                }
                break;
            case SHIELD:
                player.addShield(50);
                break;
            case SCORE:
                score += 100;
                break;
        }
    }

    private void fireBullet() {
        if (!gameStarted || gamePaused || gameOver) return;

        if (currentLevel.isLaserUnlocked() && chargingLaser) {
            // 发射激光
            if (chargeTime >= 1000) {
                bullets.add(new Bullet(player.getX() + player.width/2, player.getY(),
                        true, Bullet.WeaponType.LASER));
                chargingLaser = false;
                chargeTime = 0;
            }
            return;
        }

        // 普通武器发射
        double x = player.getX() + player.width/2.0;
        double y = player.getY();
        int weaponLevel = player.getWeaponLevel();

        switch (weaponLevel) {
            case 1:
                bullets.add(new Bullet(x, y, true, Bullet.WeaponType.NORMAL));
                break;
            case 2:
                bullets.add(new Bullet(x - 10, y, true, Bullet.WeaponType.DOUBLE));
                bullets.add(new Bullet(x + 10, y, true, Bullet.WeaponType.DOUBLE));
                break;
            case 3:
                bullets.add(new Bullet(x, y, true, Bullet.WeaponType.TRIPLE));
                bullets.add(new Bullet(x - 15, y, true, Bullet.WeaponType.TRIPLE));
                bullets.add(new Bullet(x + 15, y, true, Bullet.WeaponType.TRIPLE));
                break;
            case 4:
                // 散射模式
                for (int i = -2; i <= 2; i++) {
                    bullets.add(new Bullet(x, y, true, Bullet.WeaponType.SPREAD,
                            Math.PI / 12 * i));
                }
                break;
            default:
                // 最高等级：组合武器
                bullets.add(new Bullet(x, y, true, Bullet.WeaponType.TRIPLE));
                bullets.add(new Bullet(x - 20, y, true, Bullet.WeaponType.DOUBLE));
                bullets.add(new Bullet(x + 20, y, true, Bullet.WeaponType.DOUBLE));
                break;
        }
    }

    @Override
    public void keyPressed(KeyEvent e) {
        if (player != null) {
            player.setKey(e.getKeyCode(), true);
        }

        switch (e.getKeyCode()) {
            case KeyEvent.VK_SPACE:
                fireBullet();
                break;
            case KeyEvent.VK_SHIFT:
                if (currentLevel.isLaserUnlocked() && !chargingLaser) {
                    chargingLaser = true;
                    chargeTime = 0;
                }
                break;
            case KeyEvent.VK_P:
                gamePaused = !gamePaused;
                break;
            case KeyEvent.VK_ESCAPE:
                if (gameStarted) {
                    gameStarted = false;
                    startButton.setVisible(true);
                }
                break;
        }
    }

    @Override
    public void keyReleased(KeyEvent e) {
        if (player != null) {
            player.setKey(e.getKeyCode(), false);
        }
        if (e.getKeyCode() == KeyEvent.VK_SHIFT) {
            chargingLaser = false;
            chargeTime = 0;
        }
    }

    @Override
    public void keyTyped(KeyEvent e) {}

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Space Shooter");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setResizable(false);
            frame.add(new Game());
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}