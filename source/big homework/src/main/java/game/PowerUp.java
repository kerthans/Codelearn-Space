package game;

import java.awt.*;

public class PowerUp extends GameObject {
    private static final int SPEED = 3;
    private PowerUpType type;

    public enum PowerUpType {
        HEALTH("❤️"),
        WEAPON("⚡"),
        SHIELD("🛡️"),
        SCORE("💎");

        private final String emoji;

        PowerUpType(String emoji) {
            this.emoji = emoji;
        }

        public String getEmoji() {
            return emoji;
        }
    }

    public PowerUp(double x, double y, PowerUpType type) {
        super(x, y, 30, 30);
        this.type = type;
    }

    @Override
    public void update() {
        y += SPEED;
        if (y > 600) {
            active = false;
        }
    }

    @Override
    public void render(Graphics2D g) {
        g.setFont(new Font("Arial", Font.PLAIN, 25));
        g.drawString(type.getEmoji(), (int)x, (int)y + height);
    }

    public PowerUpType getType() {
        return type;
    }
}
