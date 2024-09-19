**1.题目要求：**  
学习成绩 > 90 分的同学用 A 表示，60-89 分之间的用 B 表示，60 分以下的用 C 表示。

**程序：**

```java
public class Main {

    public static void main(String[] args) {
        // 题目1
        int score = 55;
        char grade = (score > 89) ? 'A' : ((score >= 60) ? 'B' : 'C');
        System.out.println("学生的成绩等级是：" + grade);
    }
}
```
**程序运行截图：**
![第一题](https://github.com/kerthans/codelearn/blob/main/image/1.png)
---
**2.题目要求：**  
所谓”水仙花数”是指一个三位数，其各位数字立方和等于该数本身。例如：153 是一个”水仙花数”，因为 153 = 1³ + 5³ + 3³。

**程序：**

```java
public class sec {

    public static void main(String[] args) {
        // 2. 打印“水仙花数”
        for (int i = 100; i < 1000; i++) {
            if (isNarcissistic(i)) {
                System.out.println(i + " 是一个水仙花数");
            }
        }
    }

    // 检查是否为水仙花数
    private static boolean isNarcissistic(int number) {
        int originalNumber = number;
        int sum = 0;
        while (number > 0) {
            int digit = number % 10;
            sum += Math.pow(digit, 3);
            number /= 10;
        }
        return sum == originalNumber;
    }
}
```

**程序运行截图：**
![第二题](https://github.com/kerthans/codelearn/blob/main/image/2.png)
---
**3.题目要求：**  
判断 101-200 之间有多少个素数，并输出所有素数。

**程序：**

```java
public class third {

    public static void main(String[] args) {
        // 3. 判断101-200之间有多少个素数
        int primeCount = 0;
        for (int i = 101; i <= 200; i++) {
            if (isPrime(i)) {
                System.out.println(i + " 是一个素数");
                primeCount++;
            }
        }
        System.out.println("101-200之间共有 " + primeCount + " 个素数");

    }

    // 检查是否为素数
    private static boolean isPrime(int number) {
        if (number <= 1) {
            return false;
        }
        for (int i = 2; i <= Math.sqrt(number); i++) {
            if (number % i == 0) {
                return false;
            }
        }
        return true;
    }
}
```

**程序运行截图：**
![第三题](https://github.com/kerthans/codelearn/blob/main/image/3.png)
---
**4.题目要求：**  
有 1、2、3、4 个数字，能组成多少个互不相同且无重复数字的三位数？都是多少？

**程序：**

```java
public class four {

    public static void main(String[] args) {
        // 4. 有1、2、3、4个数字，能组成多少个互不相同且无重复数字的三位数?都是多少?
        int count = 0;
        for (int i = 1; i <= 4; i++) {
            for (int j = 1; j <= 4; j++) {
                for (int k = 1; k <= 4; k++) {
                    if (i != j && i != k && j != k) {
                        int number = i * 100 + j * 10 + k;
                        System.out.println(number);
                        count++;
                    }
                }
            }
        }
        System.out.println("共有 " + count + " 个互不相同且无重复数字的三位数");
    }

}
```

**程序截图：**
![第四题](https://github.com/kerthans/codelearn/blob/main/image/4.png)
