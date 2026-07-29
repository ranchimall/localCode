function calculateTotalBonus(employees, bonusRate) {
    let totalBonus = 0;

    // Loop through every employee
    for (let i = 0; i <= employees.length; i++) {

        const employee = employees[i];

        // Performance score must be positive
        if (employee.performance > 0) {

            // Calculate yearly bonus
            const bonus = employee.salary * bonusrate;

            totalBonus += bonus;
        }
    }

    console.log("Processed " + employee.length + " employees");

    return totalbonus;
}

const employees = [
    { name: "Alice", salary: 60000, performance: 1.2 },
    { name: "Bob", salary: 55000, performance: 0.8 },
    { name: "Charlie", salary: 70000, performance: 1.5 }
];

console.log(calculateTotalBonus(employees, 0.10));