// Level.java
package game;

public class Level {
    private int levelNumber;
    private int enemyCount;
    private int specialEnemyCount;
    private int bossHealth;
    private double enemySpeed;
    private int powerUpFrequency;
    private boolean bossLevel;
    private int targetscore;
    private int maxWeaponLevel;
    private boolean laserUnlocked;

    public Level(int levelNumber) {
        this.levelNumber = levelNumber;
        calculateLevelParameters();
    }

    private void calculateLevelParameters() {
        switch (levelNumber) {
            case 1:
                enemyCount = 10;
                specialEnemyCount = 0;
                enemySpeed = 2.0;
                powerUpFrequency = 300;
                maxWeaponLevel = 3;
                targetscore = 200;
                laserUnlocked = false;
                break;
            case 2:
                enemyCount = 15;
                specialEnemyCount = 0;
                enemySpeed = 3.0;
                powerUpFrequency = 250;
                maxWeaponLevel = 3;
                targetscore = 500;
                laserUnlocked = false;
                break;
            case 3:
                enemyCount = 12;
                specialEnemyCount = 5;
                enemySpeed = 2.5;
                powerUpFrequency = 200;
                maxWeaponLevel = 4;
                targetscore = 800;
                laserUnlocked = true;
                break;
            case 4:
                enemyCount = 20;
                specialEnemyCount = 8;
                enemySpeed = 3.5;
                powerUpFrequency = 150;
                maxWeaponLevel = 5;
                targetscore = 1200;
                break;
            case 5:
                bossLevel = true;
                bossHealth = 1000;
                enemyCount = 5;
                specialEnemyCount = 3;
                enemySpeed = 3.0;
                powerUpFrequency = 100;
                maxWeaponLevel = 5;
                targetscore = 2000;
                break;
            default:
                // 通关后的无尽模式
                enemyCount = 20 + (levelNumber - 5) * 3;
                specialEnemyCount = 10 + (levelNumber - 5);
                enemySpeed = 3.0 + (levelNumber - 5) * 0.2;
                powerUpFrequency = Math.max(50, 100 - (levelNumber - 5) * 10);
                maxWeaponLevel = 5;
                targetscore = 2000 + (levelNumber - 5) * 500;
        }
    }

    public boolean isBossLevel() { return bossLevel; }
    public int getBossHealth() { return bossHealth; }
    public int getLevelNumber() { return levelNumber; }
    public int getEnemyCount() { return enemyCount; }
    public int getSpecialEnemyCount() { return specialEnemyCount; }
    public double getEnemySpeed() { return enemySpeed; }
    public int getPowerUpFrequency() { return powerUpFrequency; }
    public int getTargetScore() { return targetscore; }
    public int getMaxWeaponLevel() { return maxWeaponLevel; }
    public boolean isLaserUnlocked() { return laserUnlocked; }
}