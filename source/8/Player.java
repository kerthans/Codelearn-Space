public class Player {
    private String name;
    private int health;
    private int money;
    private int resources;

    public Player(String name) {
        this.name = name;
        this.health = 100;
        this.money = 0;
        this.resources = 0;
    }

    public String getName() {
        return name;
    }

    public int getHealth() {
        return health;
    }

    public int getMoney() {
        return money;
    }

    public int getResources() {
        return resources;
    }

    public void takeDamage(int damage) {
        health -= damage;
        if (health < 0) health = 0;
    }

    public void heal(int amount) {
        health += amount;
        if (health > 100) health = 100;
    }

    public void addMoney(int amount) {
        money += amount;
    }

    public void addResources(int amount) {
        resources += amount;
    }

    public void useResources(int amount) {
        resources -= amount;
    }
}