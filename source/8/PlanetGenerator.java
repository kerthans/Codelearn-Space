import java.util.Random;

public class PlanetGenerator {
    private static final String[] PREFIXES = {"阿尔法", "贝塔", "伽马", "德尔塔", "艾普斯", "奥米伽"};
    private static final String[] SUFFIXES = {"星", "星球", "行星", "卫星"};
    private static final String[] TYPES = {"岩石", "气态", "冰冻", "熔岩", "丛林"};
    private static Random random = new Random();

    public static Planet generateRandomPlanet() {
        String name = PREFIXES[random.nextInt(PREFIXES.length)] + "-" +
                (random.nextInt(999) + 1) + " " +
                SUFFIXES[random.nextInt(SUFFIXES.length)];
        String type = TYPES[random.nextInt(TYPES.length)];
        return new Planet(name, type);
    }
}